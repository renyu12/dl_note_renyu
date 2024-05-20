# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional
import torch.utils.checkpoint as checkpoint

from einops import rearrange
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

import math

from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

# renyu: 基于ImageNet的预训练模型，是对应单模态视频任务的模型基础，要提前下载好放到目录下
MODEL_PATH = 'your_model_path'
_MODELS = {
    "videomamba_t16_in1k": os.path.join(MODEL_PATH, "videomamba_t16_in1k_res224.pth"),
    "videomamba_s16_in1k": os.path.join(MODEL_PATH, "videomamba_s16_in1k_res224.pth"),
    "videomamba_m16_in1k": os.path.join(MODEL_PATH, "videomamba_m16_in1k_res224.pth"),
}


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None,
        use_checkpoint=False
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states if residual is None else self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        if use_checkpoint:
            hidden_states = checkpoint.checkpoint(self.mixer, hidden_states, inference_params)
        else:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    layer_idx=None,
    bimamba=True,
    device=None,
    dtype=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    if ssm_cfg is None:
        ssm_cfg = {}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba=bimamba, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


# renyu: 这里做的还是按2D图像的Embedding，也是很简单没啥特别处理的
#        典型例子是一个16高*16宽*3通道patch展平
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, kernel_size=1, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.tubelet_size = kernel_size

        # renyu: 其实就是做了一个Conv3d卷积，以16*16*3的patch为卷积核，无填充步长不重复直接遍历整个224*224图像（相当于分块）
        #        输出维数根据model的参数配置，例如tiny是192维，就是192个卷积核，使得一个小块192次卷积后得到192长度的token
        self.proj = nn.Conv3d(
            in_chans, embed_dim, 
            kernel_size=(kernel_size, patch_size[0], patch_size[1]),
            stride=(kernel_size, patch_size[0], patch_size[1])
        )

    def forward(self, x):
        x = self.proj(x)
        return x
    
# renyu: 哈哈哈，model的类名还是写的VisionMamba没有改，应该是差不多拿来用的
class VisionMamba(nn.Module):
    def __init__(
            self, 
            img_size=224, 
            patch_size=16, 
            depth=24, 
            embed_dim=192, 
            channels=3, 
            num_classes=1000,
            drop_rate=0.,
            drop_path_rate=0.1,
            ssm_cfg=None, 
            norm_epsilon=1e-5, 
            initializer_cfg=None,
            fused_add_norm=True,
            rms_norm=True, 
            residual_in_fp32=True,
            bimamba=True,
            # video
            kernel_size=1, 
            num_frames=8, 
            fc_drop_rate=0., 
            device=None,
            dtype=None,
            # checkpoint
            use_checkpoint=False,
            checkpoint_num=0,
        ):
        factory_kwargs = {"device": device, "dtype": dtype} # follow MambaLMHeadModel
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        print(f'Use checkpoint: {use_checkpoint}')
        print(f'Checkpoint number: {checkpoint_num}')

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, 
            kernel_size=kernel_size,
            in_chans=channels, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # renyu: cls_token初始化为(1,1,token维数)的全0张量，其中token维数和设置的model一致，例如tiny是192
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        # renyu: 位置编码长度为序列长度+1，因为还有个cls_token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        # renyu: 时域位置编码长度为帧数（除以tublet_size，这里是可以多帧为一组，似乎默认为1了）
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, num_frames // kernel_size, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # renyu: 这是Mamba网络出来之后走一个带Dropout的全连接层做分类，例如k400任务就是196维的特征变为400分类
        self.head_drop = nn.Dropout(fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # mamba blocks
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba=bimamba,
                    drop_path=inter_dpr[i],
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        
        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon, **factory_kwargs)

        # original init
        self.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    # renyu: 不需要权重衰减正则化的部分，硬编码写到一个元组，处理时跳过
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "temporal_pos_embedding"}
    
    def get_num_layers(self):
        return len(self.layers)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, inference_params=None):
        # renyu: 先是调用PatchEmbed（即Conv3d）做下简单的embedding处理，然后调整下通道顺序
        #        输入的视频数据是5维张量 B-Batch Size C-Channel T-Time（帧数） H-Height W-Width
        #        例如torch.Size([128, 3, 8, 224, 224]) 对应一批64个视频（重复*2）、RGB三通道、一次8帧，高和宽都是224像素
        x = self.patch_embed(x)
        # renyu: 经过了PatchEmbed之后变为torch.Size([128, 192, 8, 14, 14])
        #        对应一批64个视频（重复*2），一个Patch变成192维的token，一次8帧，224*224的图像可分为14*14个16*16的小块
        B, C, T, H, W = x.shape
        # renyu: 调整下输入的格式，5维变为3维，torch.Size([1024, 196, 192])：
        #        第一维是一批处理的图像数 = 批*帧数 = 64*2*8 = 1024
        #        第二维是一张图像中的Patch分块数 = 高*宽 = 14*14 = 196
        #        第三维是token的维数，模型指定，tiny模型固定192
        #        所以这里其实和图像的处理是一样的，连续的多帧只不过是当做连续的token输入
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)

        # renyu: cls_token是ViT中就搞了的经典做法了，因为做分类任务不会对整个output sequence做处理，一般搞一个token的输出就可以了
        #        那搞哪个token呢？搞哪个都是不合适的，因为会有偏向认为当前token重要一些。所以干脆额外搞一个无意义的token输入去算自注意力
        #        这样这个token对应的output是和其他全部token无偏差的的关系，更适合作为特征
        #        用的时候也很简单，假设输入序列是n个token，直接在最前面（一般认为是0位置）加上这个cls_token就行，输入序列长度为n+1
        # renyu: expand cls_token张量是扩展其维度，后面两维-1,-1的意思是保持不变，而第一维和B*T一致
        #        例如原始为(1,1,192)->(64*2*8,1,192)，这里应该是已经被初始化为随机数了
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # renyu: cls_token拼接到x前面，x就增加了一项196->197，torch.Size([1024, 197, 192])，注意这还只是个临时的cls_token
        x = torch.cat((cls_token, x), dim=1)
        # renyu: 空域位置编码，大小是torch.Size([1, 197, 192])，也都是初始化为随机数了，保证每张图像中每个Patch的位置编码不一样即可
        x = x + self.pos_embed

        # temporal pos
        # renyu: 这里的cls_token处理有点麻烦，没太理清，大致记一下
        #        做的前面是单张图片中每一个搞了个cls_token，加上空域位置编码，但实际上这个还不够用
        #        我们希望要的是一个视频一个cls_token才对，还要考虑时域（例如一个视频采样了8帧），取出每个视频的cls_token
        #        torch.Size([128, 1, 192])，x删去所有单张图片cls_token，变成torch.Size([1024, 197, 192])->torch.Size([1024, 196, 192])
        cls_tokens = x[:B, :1, :]
        x = x[:, 1:]
        # renyu: 调整输入格式，按torch.Size([25088, 8, 192])，方便加时域位置编码
        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
        x = x + self.temporal_pos_embedding    # renyu: 加上随机初始化的时域位置编码，即每一个视频第1-8帧不一样
        # renyu: 这里又调整下输入格式，变为torch.Size([128, 1568, 192])的格式，即64个视频（重复*2），196*8个Patch，一个token 192维
        x = rearrange(x, '(b n) t m -> b (t n) m', b=B, t=T)
        # renyu: 最终加上每个视频不一样的cls_token，得到torch.Size([128, 1569, 192])
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_drop(x)

        # mamba impl
        residual = None
        hidden_states = x
        for idx, layer in enumerate(self.layers):
            if self.use_checkpoint and idx < self.checkpoint_num:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params,
                    use_checkpoint=True
                )
            else:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )

        # renyu: 没开fused_add_norm加速配置就直接分开做加残差和归一化
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        # renyu: 开了fused_add_norm加速配置直接调用方法一步处理，默认应该都是开的
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        # return only cls token
        # renyu: 应该是只取cls_token对应的输出
        return hidden_states[:, 0, :]

    # renyu: engine里面一轮训练中调用的，model(samples)直接调用的这里正向传播
    def forward(self, x, inference_params=None):
        # renyu: 通过Mamba提取出的特征是torch.Size([128, 192])，一批64个视频（重复*2），一个的输出特征是192维
        x = self.forward_features(x, inference_params)
        x = self.head(self.head_drop(x))    # renyu: 过一个全连接层，还做了dropout，得到torch.Size([128, 400])的k400分类结果
        return x


def inflate_weight(weight_2d, time_dim, center=True):
    print(f'Init center: {center}')
    if center:
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim
    return weight_3d


# renyu: 将预训练模型的参数加载到目标模型中
def load_state_dict(model, state_dict, center=True):
    state_dict_3d = model.state_dict()
    for k in state_dict.keys():
        if k in state_dict_3d.keys() and state_dict[k].shape != state_dict_3d[k].shape:
            if len(state_dict_3d[k].shape) <= 3:
                print(f'Ignore: {k}')
                continue
            print(f'Inflate: {k}, {state_dict[k].shape} => {state_dict_3d[k].shape}')
            time_dim = state_dict_3d[k].shape[2]
            state_dict[k] = inflate_weight(state_dict[k], time_dim, center=center)
    
    del state_dict['head.weight']
    del state_dict['head.bias']
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)


@register_model
def videomamba_tiny(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, 
        embed_dim=192, 
        depth=24, 
        rms_norm=True, 
        residual_in_fp32=True, 
        fused_add_norm=True, 
        **kwargs
    )
    model.default_cfg = _cfg()
    # renyu: 开启预训练模式则根据硬编码的地址加载预训练模型（默认都开启，除非是finetune升高分辨率的模式）
    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["videomamba_t16_in1k"], map_location='cpu')
        load_state_dict(model, state_dict, center=True)
    return model


@register_model
def videomamba_small(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, 
        embed_dim=384, 
        depth=24, 
        rms_norm=True, 
        residual_in_fp32=True, 
        fused_add_norm=True, 
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["videomamba_s16_in1k"], map_location='cpu')
        load_state_dict(model, state_dict, center=True)
    return model


@register_model
def videomamba_middle(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, 
        embed_dim=576, 
        depth=32, 
        rms_norm=True, 
        residual_in_fp32=True, 
        fused_add_norm=True, 
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["videomamba_m16_in1k"], map_location='cpu')
        load_state_dict(model, state_dict, center=True)
    return model


if __name__ == '__main__':
    import time
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table
    import numpy as np

    seed = 4217
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    num_frames = 8
    img_size = 224

    # To evaluate GFLOPs, pleaset set `rms_norm=False` and `fused_add_norm=False`
    model = videomamba_middle(num_frames=num_frames).cuda()
    flops = FlopCountAnalysis(model, torch.rand(1, 3, num_frames, img_size, img_size).cuda())
    s = time.time()
    print(flop_count_table(flops, max_depth=1))
    print(time.time()-s)

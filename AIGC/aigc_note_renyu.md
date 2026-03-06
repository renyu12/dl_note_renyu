# 概述  
## 方法  
训练时：用真实视频 -> 编码成 latent/token -> 加噪/Mask -> 模型学习如何恢复它  
### 扩散模型条件控制技术  
分3个大类：  
（1）训练时就加入条件控制  
以Conditional Diffusion为代表，用很多输入-生成对来训练  
（2）生成时调整让结果更符合条件  
以CFG（Classifier-Free Guidance）为代表，同一个模型跑两次，一次遵循条件，一次不遵循条件（如空输入），认为条件带来的变化方向就是目标方向，所以把结果做差再放大即可。之所以CFG叫做这个名字，是因为早期的思路就是有个分类器Classifier，然后看分类器梯度，向目标分类调整，但是实际应用发现额外的分类器不好做，而且对于扩散模型生成中的噪声图像本来就分类不稳定。  
（3）直接在网络结构中加控制通道  
以ControlNet为代表，在生图的时候会需要输入像素结构信息，例如姿态pose、深度图depth、边缘edge、分割图segmentation，确保生成的图会明确符合要求而不是文本描述这样过于模糊的条件，工程技术已经很成熟了。不过视频生成时域维度很难做，音频也不存在这样的空间结构，所以就是主要适用于生图的技术。  
  
## 发展  
* GAN(/VAE) 2015-2019  
GAN都比较熟悉了，经典图像生成模型，训练一个生成器输入噪声，输出图像，然后判别器看和真实图像的区别，最终直接从噪声生成图像。生成视频就是把视频当做图像序列生成，直接搞多帧，但是实际结果很不连贯，而且分辨率也只能很低  
另外还有不太主流的方法是用VAE，原理就是Auto Encoder先压缩成latent表示再重建回去，VAE是一种特殊的Auto Encoder，latent表示高斯分布形式（给出μ和σ两个feature map），所以这个时候只要单独拿出来Decoder部分，然后换着高斯分布输入就可以当生成模型用了。但是VAE的原理是学习平均图像而非精确图像，重建后纹理丢失严重，只能做模糊低质量的小图  
* Conv Diffusion 2020-2021  
把图像的扩散模型开始用于视频，3D-UNet，有点Encoder-Decoder的雏形，3D CNN可以提取feature map，效果是好一些，但是复杂度很高难以扩展  
* Latent Video Diffusion 2022-2023  
现代视频扩散模型的开端，结合了Diffusion和VAE，即缓解了diffusion计算开销大的问题，也解决了VAE重建模糊的问题。  
原理是先用Video VAE encoder，把视频Encode压缩为效果很好的latent feature map表示（比3D CNN强多了，可以做可逆的Decoder，可以看做是标准、稳定的latent分布），然后直接在latent空间扩散，获得高质量的latent特征（直接VAE做重建太糊了，但扩散模型可以补足的细节），最后用Video VAE decoder还原到像素域，就是质量较好的生成视频  
但是这里转换到latent域只是缓解diffusion复杂度问题，还是用3D-Unet Diffusion，3D卷积复杂度立方级增长（一般只能做到16x128x128这种级别），所以上限高了一些，像素域能做到32x768x768这种级别，但依然存在时长和分辨率难进一步扩展的问题  
代表工作是：  
Make-A-Video（Meta, 2022）  
Imagen Video（Google, 2022）  
Stable Video Diffusion（SVD, 2023）  
* Transformer Diffusion 2023-2024  
Transformer比UNet效果好，原生支持长时序，复杂动态，还可以多模态对齐，并且scale到更大规模，直接大大提升效果，分辨率提升到1024x1024甚至2048x2048级别，而且能够理解文本语义，细节纹理强，时间一致性好  
代表工作就是Stable Diffusion3（2024），MM-DiT + Flow Matching  
FLUX Transformer（2024）也是做的很好的工作，效率比较高  
Flow Matching适配Transformer，训练更稳定，彻底取代了传统DDPM  
* Token-based Video Generation 2024-2025  
当前商用的各种模型都是基于这一架构：  
OpenAI Sora（2024）Kling（2024–2025）Runway Gen-2.5 / Gen-3（2024–2025）Pika（2024）  
几乎解决了之前所有的核心瓶颈，包括时长（支持到数十秒甚至分钟级别）、分辨率（4K都可以做）、文本理解极强、动作场景一致（甚至切换镜头、场景都可以一致），真正可以用于商业生产了  
其实和之前没有本质区别，只是视频特征又换了个空间，核心点是不再用VAE转为latent空间，直接用Tokenizer转为token，这就更适合Transformer处理了（注意这里Token特指离散的“视觉词汇”，学术界有VQ token/Discrete Video Token等称呼，不是说ViT那种直接格式上做成patch Embedding可以输入Transformer的假patch Token）。video token同样可以decode重建，而且信息量更大（举一个具体例子，例如512x512x16帧xRGB3通道，转latent一般是64x64（空间压8倍）x4帧（时间压4倍）x4通道=65535个float32，这个输入不小，而且是Transformer不好处理的。但转token一般是16x16x4帧的patch一个token，所以就是4096个token，这个Transformer做起来很熟，有语义先验知识了）。  
然后就还是一样的用Transformer Diffusion模型对token做扩散  
注意放到Token的形式也没有进一步减少计算量，复杂度还是很大，只是可以通过复用Token的一些优化手段，并且scale加GPU来做大了  
但是暂时没有好的Token-based开源视频生成模型放出来，毕竟成本太高商业价值太大了，最强开源的腾讯混元都是Latent-based  
* 当前方向  
视频音频文本三模态生成  
世界模型包含物理、动作因果知识  
3D aware  
指定运镜  
生成控制  
长视频叙事连贯性  
  
# 商业模型使用  
可以查查关于各个制作团队的一些模型选择的分享，Artificial Analysis等网站还有一些模型的技术分排名  
## 流水线概述  
### 剧本  
现在已经完全可以个人制作完整、高质量的AIGC短片。由于AIGC视频还只是比较短的镜头，所以“拍摄”流程肯定不是长镜头的处理，而是前期先完整规划好剧本、人物设定、分镜再制作。  
作为内容产业，剧本依然是核心的部分，而且也是有门槛的部分，在网文领域已经成功的IP往往是改编成功AI剧的保证，并且直接使用小说也难以匹配节奏更快视觉更强的影视剧，还需要有编剧进行更适配的改编。  
常用工具：  
### 人设  
角色设定图（可以根据外貌、服装、性格等描述生成），还是生图工作  
常用工具：  
### 分镜  
场景示意图+运镜说明，这也是一个很有门槛的部分。  
### 资产生成  
制作首先是资产阶段。最主要的是人物部分要准备比较多，还有是分镜的关键帧用于图生视频的，需要准备好素材，  
常用工具：  
* Nano Banana  
  
### 视频生成（抽卡）  
然后是视频生成阶段。逐个镜头生成视频，可以用文生视频，有做好的图片也可以图生视频，这里是一个很大工作量和很高算力成本的部分，需要不断调整提示词，大量生成并挑选，控制好一致性和运镜。运镜部分其实文本就可以进行控制，生成模型的先验信息中已经有了大量的运镜知识。一些AIGC平台还把这一部分都封装好了，UI界面可以显式地控制。尤其是通过首尾帧补帧的图生视频模式能够控制的很精细。  
常用工具：  
* 通义万象WAN  
* 可灵  
* 海螺  
* SeeDance即梦  
* Sora  
* Vidu  
* Runaway  
### 剪辑&配音  
还涉及到一些补充工作。最主要的就是台词&口型部分，现在的处理方式一般是TTS从文本生成语音，如果模型本身支持音频驱动（如Pika、Runaway等模型支持lip-sync）就直接匹配，如果模型不支持就用Wav2Lip等模型来专门完成对嘴型的工作。  
最终是后期阶段，这一部分回归传统了，就是和普通视频制作一样整理AIGC生成的每一段素材，然后做剪辑和后期。  
  
## 人物一致性  
基础的方式是做好人物参考图，然后提取出Identity Embedding输入模型，作为一个控制条件，目前很多模型已经能够直接支持很好的一致性。  
如果要做的更好，应该是通过较多角度图片训练一个角色专属的子模型/LoRA/Identity Adapter。  
最好的效果就是直接做出数字人，而不是让视频生成模型去生成，只是做一些驱动、渲染、特效，是CG技术了。  
## 画质  
视频生成模型普遍还是256-720这个量级的分辨率，难以直接生成1080p/4k的高画质视频。  
不过整个流程目前已经比较成熟了，就是中分辨率生成+后期视频重建+（局部修复）。中分辨率生成控制算力开销，也更容易保持长时间的一致性。  
  
  
# 开源模型使用  
## VideoCrafter2  
*24.1.17 腾讯AI Lab  VideoCrafter2: Overcoming Data Limitations for High-Quality Video Diffusion Models*  
https://github.com/AILab-CVC/VideoCrafter  
比较弱的开源模型，提供了320x512/576x1024两个分辨率的I2V和T2V模型，效果测试较差，文档也不行要自己看代码使用  
```  
conda create -n videocrafter python=3.8.5  
conda activate videocrafter  
pip install -r requirements.txt  
  
# 下载T2V checkpoint  
mkdir checkpoints/base_512_v2  
# 下载HF模型折腾半天……VideoCrafter/VideoCrafter2  
  
# 参考scripts/run_text2video.sh和scripts/evaluation/Inference.py自己写启动命令  
python3 scripts/evaluation/inference.py \  
--mode 'base' \  
--ckpt_path checkpoints/base_512_v2/model.ckpt \  
--config configs/inference_t2v_512_v2.0.yaml \  
--prompt_file my_prompts.txt \  
--savedir results/base_512_v2 \  
--height 320 --width 512 \  
--fps 8 --ddim_steps 50 --ddim_eta 1.0 --unconditional_guidance_scale 12.0 --frames 16 --seed 123   
```  
  
```  
# 下载I2V checkpoint  
mkdir checkpoint/i2v_512_v1  
# 下载HF模型折腾半天……VideoCrafter/Image2Video-512  
# 还依赖CLIP的Encoder，如果下载失败要自己手动下载，然后放到下载缓存中  
# wget https://hf-mirror.com/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin      -O open_clip_pytorch_model.bin  
# /root/.cache/huggingface/hub/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/1c2b8495b28150b8a4922ee1c8edee224c284c0c/  
  
# 启动时自己准备好参考图片目录，会读取后按文件顺序一一匹配，prompt文件也是一行对应一个视频（可以设置多个）  
python3 scripts/evaluation/inference.py \  
--seed 123 --mode 'i2v' --ckpt_path 'checkpoints/i2v_512_v1/model.ckpt' \  
--config 'configs/inference_i2v_512_v1.0.yaml' \  
--savedir results/i2v_512 \  
--n_samples 1 --bs 1 --height 320 --width 512 --fps 8 --frames 16 \  
--unconditional_guidance_scale 12.0 \  
--ddim_steps 50 --ddim_eta 1.0 \  
--prompt_file my_i2v_prompts.txt \  
--cond_input my_condimage  
```  
  
## 腾讯hunyuan  
https://github.com/Tencent-Hunyuan/HunyuanVideo  
应该效果还不错，但是最低资源要求比较高，45GB显存  
  
## Wan 2.2  
  
# 论文  
有个仓库整理了一些，不过内容偏Video to Music生成  
https://github.com/Xiaohao-Liu/Awesome-Vison2Audio  
理解VTA生成模型，要关注训练数据集、位置编码设计、损失函数设计、语义&时间对齐处理  
##   
### （25.11.17MIT）Back to Basics: Let Denoising Generative Models Denoise  
He Kaiming团队备受关注的工作，思路是亮点，验证了扩散模型可以预测干净的数据x，而非当前普遍方法，预测要去除的噪声$\epsilon$或者预测流速v（v=x-$\epsilon$）。  
理论基础是基于流形假设：高维空间中的图像数据不是零散分布，而是集中依附在一个低维的流形上，这个流形是干净不包含噪声的。噪声是零散在高维空间其他部分分布的。基于这个假设，预测噪声和流速（包含噪声）在高维空间下是不好做的，需要很复杂的网络支持，训练也很困难。  
补充下流形就是一种局部像欧氏空间、整体可以弯曲的几何空间。是比较抽象，数学上比较严谨的概念，但是实际使用就理解为一个欧式空间（线段、平面、3D空间……）即可。理论上图像是256x256x3的超大高维空间，但实际分布可能只是一个低维弯曲的流形  
具体实现上用了大patch的原生Transformer（256图像用16x16,512图像用32x32），图像patch不需要tokenizer转语义域，不需要做预训练，直接输入像素块。  
还设计了一个瓶颈嵌入的优化方法，patch embedding层会先降维再升维，例如经典的768维embedding降维到16维再升回去（没看懂），但是这样强行做成低维处理，效果还不错，就进一步验证了图像分布是低维流形的假设。  
整个这一套做法极大简化了扩散模型，并且可以很好的解决高维输入难处理的问题  
  
## T2I  
### （24.3.5Stability AI SD3）Scaling Rectified Flow Transformers for High-Resolution Image Synthesis  
经典模型SD3的论文，非常有影响力。  
2个主要贡献：（1）引入并扩展 Rectified Flow（修正流）作为生成公式；（2）提出新的 Transformer 架构 MMDiT（Multimodal Diffusion Transformer），抛弃U-Net  
这两个点都成为了业界共识，是新模型设计的基本思想，不过各家闭源商业化生图模型还是都有自己设计的  
  
  
## VTA  
TODO：  
    VTA-LDM  
    V-AURA  
    MM-LDM  
    MMDisCo  
    JavisDiT++  
    SkyReels-v4  
### （21.10.17芬兰坦佩雷大学 VTA模型 SpecVQGAN）Taming Visually Guided Sound Generation  
非常早期工作，还是GAN的结构，忽略吧  
      
### （22.11.7以色列希伯来大学 Im2Wav）I Hear Your True Colors: Image Guided Audio Generation  
非常早期工作，不是VTA是ITA，不过也是会作为baseline提一下  
  
### （22.12.19人大北大-微软 T2AV模型）MM-Diffusion: Learning Multi-Modal Diffusion Models for Joint Audio and Video Generatio  
早期T2AV的经典工作，还是用的UNet而不是DiT或者audio token自回归，做了音视频两个UNet结合联合生成的设计，确实跑起来了。  
不过和25年下半年爆发的AV联合生成模型相比还是太早了，忽略吧  
  
### （23.6.29清华大学 VTA模型）Diff-Foley: Synchronized Video-to-Audio Synthesis with Latent Diffusion Models  
早期经典的VTA工作，应该是第一个把diffusion开始用于VTA任务，性能其实应该不咋地，新一点的工作不会再比较这个了，但是相比之前的SpecVQGAN和Im2Wav还是性能大大提升了，相当于把这个已经沉寂的研究方向重新拉起了  
另外也正式提出了关键挑战是时域同步和语义相关，已经在尝试用特征对齐来解决，第一步使用对比学习的音视频特征预训练（称为CAVP），损失函数设计考虑了语义和时间两方面。然后第二步使用扩散模型的时候，CAVP encoder出来的视频特征作为condition输入（我其实有点没理解，这样就能认为视频特征中包含对齐的音频信息了吗？好像也没有使用到对齐的音频特征）  
还有另外两个小的创新点，一个是数据增强，数据不够就随机切不同时长的视频片段；一个是扩散模型引导CFG+CG融合了，CFG做语义，CG做时间同步。  
Metrics也很有意思，除了用Inception Score、FID、Mean KL散度这些SpecVQGAN中用的，因为这个时候还没有视听一致性的Metrics，所以这里自己训练了一个分类器，50%是匹配的AV序列输出1，25%是有时移的，25%是不匹配的，都输出0，然后用预测的1占比作为准确度  
  
### （23.8.18悉尼大学 VTA模型）V2A-Mapper: A Lightweight Solution for Vision-to-Audio Generation by Connecting Foundation Models  
VTA中期研究时期一个可行的简单思路，尽量复用现有视频Encoder和音频生成模型，只训练一个视频latent到音频latent的映射模块。  
CLIP (视觉表征基础模型) -> V2A Mapper（唯一训练模块）-> CLAP embedding（和Prompt文本等价的输入） -> AudioLDM（经典T2A音频生成基础模型）  
理论上训练所需的数据量没那么大，论文中只用了VGGSound。  
代码没开源。  
  
### （23.9.19Meta VTA模型）FoleyGen: Visually-Guided Audio Generation  
早期VTA工作，尝试的思路是基于Audio LM做生成（把音频用类似文本的处理，转为token然后生成下一个token，再decode回去），视频特征加入作为condition，应该不是很好的技术路线，后面都是diffusion模型了，这个比V2A-Mapper引用还低。所以虽然是Meta发布，但后面很少被提到，影响力较小  
Metrics也是早期FVD、KL，不过似乎是首次将ImageBind余弦相似度拿来做视听一致性Metrics的尝试，也没有做很多解释，就说了可以表示一致性。  
另外也做了主观实验打整体质量分数、内容相关性分数、时间一致性分数，这也是和我们的思路很一致了  
  
### （24.2.27港科大 多功能AV生成模型）Seeing and Hearing: Open-domain Visual-Audio Generation with Diffusion Latent Aligners  
很有意思的工作，根本没有训练模型，提出了一个聪明的工程做法，直接复用T2V和T2A diffusion模型，然后用ImageBind Embedding来指导同时生成。  
具体做法应该是扩散模型生成音频和视频latent的时候，每一步会decode出来一下当前的音视频，然后转ImageBind embedding来分析余弦相似度，这个结果再用于diffusion sampling过程中latent更新的损失函数，也就是扩散过程中要对齐ImageBind相似度  
这个工作应该算是把ImageBind至少在评估侧的地位确立了，虽然实际效果不好，毕竟只能整体语义对齐。  
  
  
### （24.6.1浙大 VTA模型）Frieren: Efficient Video-to-Audio Generation Network with Rectified Flow Matching  
这个工作偏扩散模型底层一些，是基于Rectified Flow Matching来做VTA，并且在音视频同步上也做了比较多的channel-level扩模态融合的设计。  
Rectified Flow Matching这个我不太懂，有机会看下数学细节，大致是Flow Matching的一种变体，路径会接近直线，提升生成效率，所以Frieren模型的生成速度是更快的。我研究了下说这个和SD3的Flow Matching training还不一样，SD3只是训练loss使用flow matching，Frieren是纯flow generative模型，基于求解ODE方程不走扩散。好像还是挺硬核的。  
不过这个工作仍然显得比较特殊，后续的工作依然还还是走的基于主流DiT去魔改的路线，没有走这个Flow Matching模型的路线。  
Metrics也用了Diff-Foley的ACC Align模型，并且在AMT上做了众包主观实验打音频分和内容一致性分数，附录里还给了点介绍。不过应该是做的很简单，就是一个音频6个人打分，时薪8美金，然后是1-5分段，0.5步长选一个，这是个扩大规模的好做法。  
  
### （24.6.11多伦多大学 T2AV模型）AV-DiT: Efficient Audio-Visual Diffusion Transformer for Joint Audio and Video Generation  
第一个基于DiT的联合AV生成模型。可以直接从图像DiT扩展到音频生成是很强的工程Insight。  
模型设计上应该还是基于图像DiT去修改，尽量少做改动，更多是偏adapter的设计，时域adapter使得图像DiT可以处理视频，音频adapter使得图像DiT可以处理音频，Fusion Adapter使得音视频可以相互作用（用的联合起来自注意力而不是交叉注意力，说参数更少一些）。训练的时候也是冻结了很多预训练的图像DiT模块。不过我看了下具体结构图，感觉设计还是有点点复杂的，音视频的2个分支是有差异的，然后有些交互。  
Metrics用的很传统的做法，值得考虑下。只考虑了视频和音频质量，并且是通过latent特征分布来分析的，在Benchmark数据集上分析视频的FVD、KVD，音频的FAD，都是分布指标。（这些指标应该新的工作中不再使用了）  
早期工作没有特别做显式的视听同步，所以效果也是没那么好的。  
  
### （24.6.17ElevenLabs）VTA API  
ElevenLabs是在做AI语音方向很强的一家公司，主要的业务还是  
AGAV-Rater论文中测试了ElevenLabs的VTA模型，但是他们没有做相关技术论文，甚至我没查到有VTA模型，研究了下看到有提供一个VTA的API  
https://github.com/elevenlabs/elevenlabs-examples/tree/main/examples/sound-effects/video-to-sfx  
但是这个API实际解释了原理是Video->抽帧->发到GPT-4o视觉理解，生成 SFX 文本描述->发到Text to Sound模型生成音效  
所以这个模型是没有时间同步的，不是VTA模型。  
不过工业界早期这样的做法也是很合理，因为VTA模型是数据需求量大成本高而商业价值又低的一个场景，没有语音赚钱。  
  
### （24.7.1上海AI Lab VTA模型）FoleyCrafter: Bring Silent Videos to Life with Lifelike and Synchronized Sounds  
和V2A-Mapper路线比较接近的工作，复用冻结的T2A生成模型，但做的灵活了很多，支持更复杂的condition控制，实现更好的一致性效果。  
按说T2A生成模型只能输入一路文本condition，但是FoleyCrafter改进支持了3路输入，一是文本Prompt（可选），二是视频语义信息，加了个可训练的视频语义适配器模块，三是视频时间信息，加了可训练的视频时间戳检测器和时间适配器模块。  
语义分支是和文本Prompt分支交叉注意力融合在一起注入U-Net各层的，时间分支特别一点，加到U-Net的feature map中（说类似ControlNet的思路）。  
具体实现和训练感觉是比较复杂的，论文中汇报比V2A-Mapper有提升，但是也看到后续论文如AudioGen-Omni显示没提升。我理解复用T2A确实不是最优的方案，文本对于空间、时间同步的控制很有限。  
  
### （24.7.8韩国NAVER实验室 ReWaS）Read, Watch and Scream! Sound Generation from Text and Video  
TODO: 走的不太主流的技术路线：T2A模型通过Video做条件，但是条件用的很有思路，考虑了时间维度的能量变化，应该是个挺有意思的工作  
Metrics也没啥特别的，对比Seeing&Hearing性能有所提升（估计比不过主流路线的工作）  
  
### （24.11.8华盛顿大学 VTA模型 VATT）Tell What You Hear From What You See -- Video to Audio Generation Through Text  
主要做的是TVTA，也就是支持VTA的时候可选文本condition  
做法大致是一阶段先训练用一个LLM可以输入视频生成可能的音频Caption，二阶段就使用这个LLM生成audio token（或者使用自己定义的文本Prompt转audio token），输入AudioLM生成音频，感觉有点V->T->A的意思，虽然不是显式的，也不是主流路线，效果应该不行。  
Metrics中视听一致性用的同diff-foley的Align Acc指标。  
  
### （24.12.19UIUC VTA模型）MMAudio: Taming Multimodal Joint Training for High-Quality Video-to-Audio Synthesis  
VTA现在三模态联合训练范式的经典工作，github 2k+星，工程化实现做的很不错。  
之前最经典的VTA模型训练分两种范式，一是只用视频&音频两个模态，从头开始训练，但是因为视听数据集太少（不是随便视频数据集就可以，VTA主要是配sound，但是实际的数据集中大量的是speech和music，还有很多后期额外声音，导致没法直接用）；二是更主流的，用TTA的模型（数据集很多）再额外用视听数据集训练控制模块，网络会比较复杂（说的就是FoleyCrafter！），性能没有三模态好。  
理论上就是要三个模态一起“对齐”才好，但是没有足够的AV、VAT数据支持怎么办？关键在于用好缺失模态的数据支持Missing Modality Training，AT是有大规模开源数据的（主要是WavCaps的7600小时），然后加上小规模的AV数据（就是VGGSound的500+小时，理论上VAT、VT也有一些不过没使用）。事实证明这样效果也不错，多模态生成模型需要的生成能力，不要求多模态显式对齐。  
训练的时候就是统一的三模态输入，预测音频latent表示的flow/velocity。（AIGC音视频图像这些连续信号生成任务和MLLM主流路线（算是离散token生成）不一样，不一定依赖对比学习显式地三模态特征对齐啥的，可能会用CLIP等Encoder。可以理解为信号形式本身都有梯度可以算loss对齐，不需要对比学习这种额外的loss）输入AT双模态数据的时候Video用empty的token代替。  
损失函数就是Conditional Flow Matching。  
但是直接这样训练是做不好时间对齐的，这也是VTA的难点所在，论文中也是做了比较好的设计，做了个条件同步模块，引入了frame-level conditioning，使用了SynchFormer特征，会把音频事件和具体哪一帧对应。  
  
### （25.2.6中国电信 多功能AV生成模型）UniForm: A Unified Multi-Task Diffusion Transformer for Audio-Video Generation  
好像不是很强的工作，算是早期T2A+A2V+T2AV多功能的工作，主要的贡献就是实现一个模型多任务，原理是通过不同的噪声输入，配合不同的任务Embedding（类似CLS加一个额外的token标记是VTA、ATV还是T2AV）。效果说对比各自单任务模型效果都还是不错的，但是可能没有其他创新性的设计，影响力较小。  
做了个demo页面但是没有开源代码https://uniform-t2av.github.io/  
  
  
### （25.3.30浙大 T2AV模型）JavisDiT: Joint Audio-Video Diffusion Transformer with Hierarchical Spatio-Temporal Prior Synchronization  
主要做的是改进的DiT模型，设计了新的DiT block中，使用时空自注意力、粗粒度交叉注意力、细粒度交叉注意力、音视频双向注意力和FFN层  
做了一个新的Metrics叫做JavisScore，不是直接用整段的语义一致性，而是切重叠2s小段，算ImageBind相似度，并且只取其中40%最不同步的帧，认为这样的方式会更符合主观感知，明显不同不变的片段影响很大，不要被大部分同步部分给平均弱化了。这个思路很好。  
还做了一个10140个多样化视频的Benchmark，JavisBench。（我有点不太理解，T2AV模型建个Benchmark也没啥用，还是一些NR指标，只是可以把Prompt/Caption分类了）  
  
### （25.6.24快手 VTA模型）Kling-Foley: Multimodal Diffusion Transformer for High-Quality Video-to-Audio Generation  
是一个集大成的拼装模型，稍有点复杂。主要是基于SD3和MMAudio的工作去优化。  
* Encoder部分  
首先说Encoder框架是参考SD3 (Stable Diffusion 3，扩散模型经典baseline)框架做修改的，文本Encoder是T5-Base，视觉Encoder用的MetaCLIP，还有个对齐Encoder用的Synchformer（这个稍有点不好理解，输入是视频抽出来的一些图像帧，输出是可以用于时域对齐的信息）  
* Decoder部分  
还是参考SD3框架的MM-DiT Decoder框架，不过做了一些改进，扩展了时序同步模块，使用了动态掩码策略。音视频token使用基于RoPE的时间位置编码，能够更好对齐。  
为了实现可变长序列生成，引入了可学习的时长embedding  
* 损失函数部分  
使用Flow Matching的概念（这个是数学上的一种分布对齐方法，学习从一个分布到另一个分布的光滑映射，原理是学习一个常微分方程dx/dt = v(x, t)，使得x(0)是初始分布，x(1)就是目标分布），用到扩散模型中，就是常用的学习速度向量场的模式（用x(0)代表完全的高斯噪声，x(1)代表干净的原始图像，整个过程就是噪声流向数据的连续变换，可以预测v而不是噪声$\epsilon$也可以时间），这个速度向量场的含义大概是在t时间，噪声如何向真实数据移动。  
这已经是扩散模型的通用训练范式了，代码也都是封装好的，参考两篇经典论文：  
Flow matching for generative modeling  
Improving and generalizing flow-based generative models with minibatch optimal transport  
ChatGPT还提到Flow Matching可以用来替代CLIP对比学习方法做特征对齐，是未来的方向，有点意思  
* 位置编码处理  
为了时间上对齐音频和视频，不能直接各自做RoPE，需要考虑尺度。可以发现音频的Token是更密的（音频约10ms一个token，视频约40ms一个token），所以这里将视频Token做RoPE的角度做了缩放（可以理解为调整频率对齐）  
注意仅仅靠这个时间轴对齐还是不够的，后面还是要靠对齐Encoder来进一步提取对齐特征  
  
为什么SOTA的VTA模型还是用的VAE到latent空间而不是做Token？这里应该是因为这是音频生成任务，而不是长视频生成，latent特征已经够用了。另外大的Token-based模型可能不开源也用不了  
  
代码和模型似乎没有开源，关注度不高。同时还做了一个Benchmark Kling-Audio-Eval，精心筛选了20,935个以音效为主的视频，手动注释音频caption、视频caption、声音事件，涵盖九种声音场景1919个细分类别，这个是开源的。  
  
### （25.6.26阿里通义 VTA模型）ThinkSound: Chain-of-Thought Reasoning in Multimodal Large Language Models for Audio Generation and Editing  
引入MLLM CoT来做VTA的条件控制，把黑盒的生成任务转变为可解释的推理任务，很有思路的工作。  
把生成划分为三个阶段：  
（1）基础Foley生成，创建语义和时间匹配的全局声场;  
（2）通过用户点击进行基于区域的互动细化;  
（3）基于高级指令（Add / Remove / Inpaint）进一步做音频编辑。  
在每个阶段，MLLM生成结构化的CoT推理，指导统一的音频生成模型来生成和编辑音轨。音频生成模型是优化版的MM-DiT，可以输入CoT推理格式的条件控制信息。  
TODO  
为了能够适配生成模型，MLLM输出不能只是随便的Prompt，而要是结构化的序列  
  
### （25.8.1快手 全能TA模型）AudioGen-Omni: A Unified Multimodal Diffusion Transformer for Video-Synchronized Audio, Speech, and Song Generation  
不是一个单纯的VTA模型（一般指输入视频+可选文本，生成环境声Foley），而是一个支持任意模态输入（视频/音频/描述文本/歌词or字幕），生成任意类型声音（Speech/Music/Foley）的全能模型，直接把这个任务扩展了，确实是未来的方向。升华一点说，这是audio的GPT，用audio替代文本成为了基础模型的核心模态。  
当然创新点主要在多模态任务的拓展而不是网络设计，模型结构上有一些改动，但是基本还是MMAudio：  
* PAAPI（Phase-Aware Positional Encoding）  
做RoPE位置编码的时候，视频、音频、歌词or字幕会加时间RoPE信息（但是描述文本模态不会）  
* 解冻所有模块  
Video encoder、Audio encoder、Text encoder、Fusion transformer、Diffusion head都训练  
* 统一的Audio Generation Head  
不区分Speech、Music、Foley  
做的好的点还是Missing modality pretraining，从而可以实现任意模态输入，适配各种任务  
  
  
### （25.8.23腾讯 VTA模型）HunyuanVideo-Foley: Multimodal Diffusion with Representation Alignment for High-Fidelity Foley Audio Generation  
比较平实的工业界VTA模型报告，应该是对标Kling-Foley开源的，架构同样是MMAudio优化。虽然没有直接对比Kling-Foley，但是测了Kling-Audio-Eval，对比之前模型是SOTA水平，有一定关注度，github千星。  
这个工作最强的点在于数据，做了一整套TV2A数据的pipeline，可以自动筛选（基础属性、切片、AudioBox质量评估，IB+DeSync达到阈值）、标注，搞了100k小时的训练数据（VGGSound才550小时）。  
文中还提到TV2A中的问题是文本Prompt模态过强，但是视频和音频的同步没那么好，还有音频生成保真度也不好。所以架构和训练上也做了改进：RoPE位置编码时，对于AV输入是交错而不是分开的（好像是挺主流的做法了）；跨模态融合用的两阶段处理，AV是frame级别的交叉注意力，然后整体AV和T是全局的交叉注意力（这个也是主流共识了，AV时间对齐会更好）；损失函数不是仅仅diffusion输出噪声预测loss，还加了个中间层hidden loss，让diffusion Transformer的中间层hidden对齐预训练的音频embedding，这个叫做Representation Alignment（REPA，24年经典diffusion改进方法，思想是diffusion不光要结果一致，中间过程也要和encoder对齐，让生成轨迹经过正确语义流形） ；解码器DAC（音频token->音频波形）也不用离散向量量化，而是用的DAC-VAE，连续表示质量更好。  
做了消融实验证明了各个优化都是有效的。  
模型结构上稍微改动MMAudio，输入文本Encoder用的CLAP、视频用的SigLIP、音频用的DAV-VAE Encoder（仅训练时有音频输入）；中间的diffusion denoiser网络就是普通的多层多模态Transformer Block，然后多层音频单模态Transformer Block，得到音频latent表示；最后输出是过DAC-VAE Decoder  
  
### （25.9.7港科广 多功能AV生成模型）UniVerse-1: Unified Audio-Video Generation via Stitching of Experts  
挺有意思的做T2AV+VTA+ATV统一模型的思路，不知道算不算是带起Joint AV Generation的先锋工作，看时间确实可以。  
是缝合很强的视频生成模型（用的WAN2.1）+音频生成模型（用的Ace-step），让这两个模型各层Transformer Block互相交叉注意力注入，例如视频生成模型video hidden作为Q，然后KV是video hidden+audio hidden，反过来音频生成模型也一样。  
由于音视频模型的latent是不对齐的，直接连确实不行，所以还是需要大量训练的，不过说会比新训练模型更稳定。也是做了一个自动标注的pipeline，然后搜集了7600小时的训练数据（工业界不算多，学术界已经相当强了）。  
还做了个小的Benchmark，其实就是600个图文对用于生成AV的。然后统计很多客观指标用于评估，包括MANIQA、MUSIQ、AudioBox、SyncNet、SynchFormer等等，可以学习下。  
这种缝合音频+视频生成模型训练的思路，应该说是介于MMAudio整套模型和V2A-Mapper简单mapping之间的，训练难度介于中间，但理论性能上限应该也介于中间。  
  
### （25.9.30Character AI 多功能AV生成模型）Ovi: Twin Backbone Cross-Modal Fusion for Audio-Video Generation  
这个工作是工业界很强的一个Joint AV Generation工作，看来同期Joint AV Generation已经要火起来了，似乎是一个技术发展必然趋势。  
训练数据量级和模型结构都是要比UniVerse-1更强的，没有复用现成的音频FM，而是音视频都用完全同构的DiT模型（可以称为Twin），视频就用现成FM，音频则从头训练，这样比随便找不成对的FM对称性更好。  
模型结构和UniVerse-1有点类似，也是音频和视频生成的分支每层Transformer Block交叉注意力交互。再拿大量AV数据训练，  
  
### （25.10.19pub ICCV 人大 VTA模型）VAFlow: Video-to-Audio Generation with Cross-Modality Flow Matching  
https://vaflow.github.io/demo/，提供了大量对比case，但似乎一直未开源  
是一个概念性的工作，核心点就是做diffusion的时候，不是从noise开始，这在VTA任务中不是最优的，忽略了视频的条件信息。  
所以设计了起点改为视频的latent再经过一个alignment VAE映射到音频latent空间再sampling（alignment VAE是要单独训练的模型，类似一个特征对齐模型，只不过VAE是对齐分布，这样可以做sampling有随机性）。  
这样的改进可以利用音视频关联的先验信息，效果会更好。  
  
### （25.11.5南京大学-腾讯 多功能AV生成模型）UniAVGen: Unified Audio and Video Generation with Asymmetric Cross-Modal Interactions  
和Harmony的同期同组（腾讯混元）工作，但是重点在于统一AV生成框架，偏工程，通过多阶段多任务的训练实现。没有像Harmony那样在视听同步上做很多工作。另外是主要针对语音的，没有提到音乐，环境声应该是支持但不是主题。  
支持任务：  
（1）Joint Audio-Video Generation  
输入文本，可选图片，同时生成音视频。  
还有个进阶玩法，输入文本、图片、音频，同时生成音视频，这样可以控制说话人音色timbre。做这个任务还有个模型设计，就是reference audio不参与cross interaction，避免音色被破坏  
（2）Joint AV Continuation（条件续写，时序延续）  
输入音视频，同时生成后面的音视频  
（3）单模态驱动生成  
VTA，文本和参考音色都可选，语音、环境声都可以生成（好像没提音乐）  
ATV，需要输入文本，生成匹配表情、动作的视频（环境声可以吗？）  
因为是针对语音视频，所以音视频一致性的评估指标是用SyncNet的Lip Sync指标，另外还自己设计了音色一致性+情绪一致性指标，这个是让Gemini 2.5打分的，具体可靠性存疑……  
  
  
### （25.11.26上交-腾讯 多功能AV生成模型）Harmony: Harmonizing Audio and Video Generation through Cross-Task Synergy  
做的是视听同步的联合音视频生成模型——Joint AV Diffusion（同时文本生成音视频）和VTA、ATV都能做，先整理下3个任务：  
（1）Joint AV Diffusion：输入文本，同时生成音视频。输入Diffusion模型的是audio latent noisy和video latent noisy，会一起  
（2）VTA：输入视频（文本可选作为条件），生成音频  
（3）ATV：输入音频（文本可选作为条件），生成视频  
文中提到了要解决的核心问题，相对而言，VTA和ATV做好时间同步都简单一些，Joint AV Diffusion很难做好时间同步  
文中总结3个Joint AV Diffusion做不好时间同步的原因：  
（1）Correspondence Drift（对应漂移）：音频视频都是在去噪生成的，每一步都在便宜，没有稳定且清晰的另一个模态信息作为参考，这是核心问题  
（2）Local vs Global 对齐冲突：定义嘴型同步这种是Local对齐问题，情绪、环境声是Global对齐问题，典型模型（MM-Diffusion, Ovi, UniVerse-1, JavisDiT等）在视频分支和音频分支输入时，是取得各自全局tokens做交叉注意力，难以同时做好Local和Global对齐  
（3）CFG 不会增强跨模态对齐：CFG (Classifier-Free Guidance)是在扩散模型中“强化条件控制”的技术，生成一版无prompt控制的，一般有prompt控制的，二者差值就可能是“正确的方向”，所以扩散去噪的时候更偏向这个方向一些。但是CFG实现的只是更靠近文本prompt，并不是音视频更同步。  
所以Harmony模型在做的时候做了3个改进：  
（1）训练阶段 —— 三个任务同时训练  
既然做Joint AV Diffusion会Correspondence Drift，那就和VTA、ATV一起训练，结果会稳定一些。  
（2）模型结构 —— 分开Global-Local模块  
Local模块做短时间对齐，Global模块做整体语义对齐  
（3）推理阶段 —— SyncCFG  
这个是个挺创新的点，想到从改进CFG的角度来增强AV同步。实现也很简单，就是不再是作差有文本Prompt和空文本Prompt两种情况，而是作差Joint AV和单模态输入AV生成两种情况，很合理，语义Prompt不再是guidance的方向了  
不过说是开源，看github仓库里一直都还没上传模型和代码，需要观望。  
  
### （25.12.15字节 多功能AV生成模型）Seedance 1.5 pro: A Native Audio-Visual Joint Generation Foundation Model  
这个技术报告发布时间也是25年底了，可能和Seedance2会比较接近。  
做的多任务，主要是T/I/V/AV输入都可以获得AV输出。  
模型结构没有细致描述，说基于MMDiT的结构，应该没有太大改动的处理，大部分篇幅还是在介绍思路和评测结果  
训练部分的思路也是多阶段渐进式的，先做T2V和T2A，然后再T2AV，最后还有AV2AV（编辑），这样会比较稳定。训练数据集情况也没有详细介绍，只是说建立了数据pipeline然后获取了高质量的AV数据。  
评估部分做的是很完整的，搞了配套的SeedVideoBench-1.5评估框架，非常值得借鉴：  
TODO  
* 视频：motion质量（分为人物动作和镜头运动），Prompt following，视觉审美？  
* 音频：Prompt following，声学质量，音画同步，audio expressive（这个感觉和我提的语义一致性比较接近，包括BGM适配性、语音的情感&语调，音频沉浸感和连贯性等）  
  
但是我不知道这么多维度的指标具体是怎么做的？  
还有主观实验评价，除了做1-5绝对评分，也用了双刺激的比较实验，分GSB（更好-一样-更差），比AB Preference可能更合理，不过也会有大量的same。  
测试结果比较诚实，很多分数不如veo 3.1都说了，kling也很强，但是会分一些具体场景分析，说明中文对话、复杂镜头、歌剧等场景有突出优势  
  
### （26.1.6lightricks T2AV模型）LTX-2: Efficient Joint Audio-Visual Foundation Model  
和Harmony是同期同类型工作，但是看起来好像差别挺大的。LTX-2感觉更偏工程落地的模型设计，Harmony的改动点novelty更大。不过LTX-2是工业界开源，有不少star，社区讨论度也更高，实用性更好的。  
创新点说是AV joint generative backbone这个似乎看到Harmony已经不新了，音视频latent分开我觉得也是很直接的思路（咋能不分开呢？好像很早期工作有吧），没啥特别的，不过强调了视频更复杂，所以视频分支做的更复杂，音频分支更轻量。  
网络结构上设计应该是比较好的，全层级 AV Cross-Attention 耦合。也重新设计了CFG（可能和Harmony思路差不多？）。  
真正强的点在于efficient，对比WAN推理速度加速18x，推理加速做了非常多优化可以学习，不愧是工业界出品。  
  
### （26.1.7快手 多功能AV生成模型）Apollo: Unified Multi-Task Audio-Video Joint Generation  
最开始26.1.7发的一版主体是Kling团队，然后命名是Klear，但是26.1.13重新上传了一版，移除了一个作者也删去了Kling团队标识，更名为Apollo，可能是涉及到一些内部问题吧  
似乎是没有开源  
做的模型是多任务的，支持文本和图像的输入，T2AV、TI2AV、T2V、T2A、TI2V都可以做  
认为AV联合生成中音画不同步、单模态生成质量下降的问题和模型结构、训练方法、数据规模有关，所以三方面都有创新：  
（1）模型结构上认为Single Tower是优于Dul Tower的，也就是音视频token是输入同一个MM-DiT Transformer的，这样注意力可以同时覆盖音视频token和各自的caption token，比双分支做一些交叉注意力能实现更强的Interaction和synchronization  
（2）训练方法上认为只训练T2AV任务不够，各种生成任务要一起来，随机mask掉音视频的token，这样训练出的模型不光能做多任务而且效果更好  
（3）训练数据集上，构建了一个81M的数据集（未公开），有自动化筛选的pipeline保证数据质量，用大模型来标注caption  
Metrics上是音视频质量、TTS质量还有视听一致性，其中视听一致性使用SynchFormer、SyncNet和IB-Score  
说性能是优于UniVerse-1、Ovi、JavisDiT，接近Veo-3的  
  
### （26.2.9OpenMOSS IT2AV模型）MOVA: Towards Scalable and Synchronized Video-Audio Generation  
能拿出来开源的AV联合生成模型，看开源仓库做的非常好。  
https://github.com/OpenMOSS/MOVA  
做的主要是IT2AV任务，其中T是辅助控制的，没有去做多任务。  
经典音视频dual tower结构，并且基于两个预训练模型降低训练量：视频是WAN2.2 I2V 14B模型，音频是1.3B的T2A Diffusion模型，两个分支通过交叉注意力bridge模块连接。有一个比较重要的改进点是特征输入时时间对齐，考虑到音频token密集，视频token稀疏，为了交互时更好对齐，使用了RoPE positional encoding，时间对齐之后再做  
评估工作做的很不错，虽然客观指标还是IB DeSync LSE-D LSE-C这一套，但是作者直接指出当前objective metrics很大局限，所以做了Arene-based主观实验（类似Chatbot Arena评测，让人来做两两比较的投票，A/B preference，ChatGPT常用），相同输入的4个模型结果两两随机对比，最后收集了5000+投片，确认  
性能不错，测试优于LTX-2、Ovi和WAN2.1+MMAudio。并且这种评估最后还可以按国际象棋类似算法给出ELO量化评分。这个方法很有意思值得借鉴。
renyu的VideoMamba源码阅读笔记
VideoMamba的代码结构基本参考Vision Mamba再参考Facebook DeiT源码，这个代码整体质量还是很高的
但改动成VideoMamba后有很多不再使用的冗余部分，并且加入了自蒸馏操作还有一系列下游任务分出了多份代码，使得阅读代码会有一定难度，推荐熟悉DeiT代码结构之后再阅读，可以对比看修改的部分

参考VideoMamba整体的代码目录整理如下：

    .
    |-- LICENSE
    |-- README.md                                # renyu: VideoMamba项目说明文档
    |-- assets                                   # renyu: 资源目录，放一些文档图像
    |   |-- comparison.png                           # renyu: VideoMamba和TimeSformer对比图
    |   `-- framework.png                            # renyu: VideoMamba架构图
    |-- causal-conv1d                            # renyu: Mamba作者Tri Dao写的一个CUDA 1D Causal depthwise卷积的库，Mamba中用到了（略，详见Mamba代码注释）
    |-- mamba                                    # renyu: Mamba模型源代码，VideoMamba使用Mamba Block搭建模型，唯一的改动是在其中加入双向扫描（略，详见Mamba代码注释，TODO：双向扫描注释）
    |-- requirements.txt                         # renyu: 依赖包目录
    `-- videomamba                               # renyu: VideoMamba主体代码目录
        |-- README.md                                # renyu: VideoMamba代码使用文档
        |-- image_sm                                 # renyu: 图像任务VideoMamba模型
        |   |-- MODEL_ZOO.md                             # renyu: 预训练模型文档
        |   |-- README.md                                # renyu: 图像任务VideoMamba模型文档
        |   |-- augment.py                               # renyu: 数据增强，随机对图像裁剪、变换并归一化等
        |   |-- datasets.py                              # renyu: 加载数据集代码，主要是ImageNet
        |   |-- engine.py                                # renyu: 核心代码，实现一轮训练的流程&评估流程
        |   |-- engine_distill.py                        # renyu: 核心代码，实现蒸馏模式一轮训练的流程&评估流程（就是输入两个模型，损失函数改为参考teacher输出）
        |   |-- exp
        |   |   |-- videomamba_small                     # renyu: 最基本的small模型参数量26M，后面middle和base都是基于small去自蒸馏的
        |   |   |   |-- run224.sh                            # renyu: 224*224分辨率的模型是最基本直接训练的，后面448和576的都是微调224得到的
        |   |   |   |-- run448.sh
        |   |   |   `-- run576.sh
        |   |   `-- videomamba_tiny                      # renyu: 更轻量的模型参数量7M，性能也不错
        |   |       |-- run224.sh                            # renyu: 224*224分辨率的模型是最基本直接训练的，后面448和576的都是微调224得到的
        |   |       |-- run448.sh
        |   |       `-- run576.sh
        |   |-- exp_distill
        |   |   |-- videomamba_base                      # renyu: 论文做的最大的模型98M参数量
        |   |   |   `-- run224.sh                            # renyu: 224*224分辨率的模型基于small自蒸馏得到，但模型大了开始过拟合，没有再去微调高分辨率了
        |   |   `-- videomamba_middle                    # renyu: 性能最好的middle模型74M参数量
        |   |       |-- run224.sh                            # renyu: 224*224分辨率的模型基于small自蒸馏得到，后面448和576的都是微调224得到的
        |   |       |-- run448.sh
        |   |       `-- run576.sh
        |   |-- generate_tensorboard.py                  # renyu: 读取模型的日志，然后画tensorboard图表展示效果的脚本
        |   |-- generate_tensorboard_distill.py          # renyu: 读取模型的日志，然后画tensorboard图表展示蒸馏效果的脚本
        |   |-- hubconf.py                               # renyu: 应该是模型发布到PyTorch Hub的配置文件，但这里没啥东西只有几个依赖包，应该是DeiT用的
        |   |-- imagenet_dataset.py                      # renyu: 没有用torchvision的ImageNet加载库，自己另外弄的，可能训练集测试集格式处理不一样
        |   |-- losses.py                                # renyu: DeiT做蒸馏的时候用到的蒸馏损失函数，VideoMamba自蒸馏没有用，直接交叉熵
        |   |-- main.py                                  # renyu: 非蒸馏模式训练&评估主函数
        |   |-- main_distill.py                          # renyu: 蒸馏模式训练&评估主函数，主要区别是初始化了student、teacher两个模型
        |   |-- models
        |   |   |-- __init__.py
        |   |   |-- deit.py                              # renyu: DeiT模型，作为Baseline比较
        |   |   |-- videomamba.py                        # renyu: VideoMamba模型
        |   |   `-- videomamba_distill.py                # renyu: 蒸馏模式VideoMamba模型，主要区别是多一个初始化参数区分是student/teacher模型，output head不同
        |   |-- run_with_submitit.py                     # renyu: 按非蒸馏模式用submitit库向服务器提交任务的代码，middle和base的224模型使用（tiny和small模型直接srun提交了）
        |   |-- run_with_submitit_distill.py             # renyu: 按蒸馏模式用submitit库向服务器提交任务的代码，middle的448、576模型使用
        |   |-- samplers.py                              # renyu: DeiT中引入的一个repeated augment数据增强方式用到的数据采样器
        |   `-- utils.py                                 # renyu: VideoMamba使用的一些其他组件
        |-- video_mm                                 # renyu: 多模态视频任务VideoMamba模型
        |   |-- DATASET.md
        |   |-- MODEL_ZOO.md
        |   |-- README.md
        |   |-- configs
        |   |   |-- beit-base-patch16-224-pt22k-ft22k.json
        |   |   |-- config_bert.json
        |   |   |-- config_bert_large.json
        |   |   |-- data.py
        |   |   |-- model.py
        |   |   |-- pretrain.py
        |   |   |-- qa.py
        |   |   |-- qa_anet.py
        |   |   |-- qa_msrvtt.py
        |   |   |-- ret_anet.py
        |   |   |-- ret_coco.py
        |   |   |-- ret_didemo.py
        |   |   |-- ret_flickr.py
        |   |   |-- ret_msrvtt.py
        |   |   |-- ret_msrvtt_9k.py
        |   |   |-- ret_msrvtt_mc.py
        |   |   |-- ret_ssv2_label.py
        |   |   `-- ret_ssv2_template.py
        |   |-- dataset
        |   |   |-- __init__.py
        |   |   |-- base_dataset.py
        |   |   |-- caption_dataset.py
        |   |   |-- dataloader.py
        |   |   |-- qa_dataset.py
        |   |   |-- sqlite_dataset.py
        |   |   |-- text_prompt.py
        |   |   |-- utils.py
        |   |   `-- video_utils.py
        |   |-- exp_pt
        |   |   |-- videomamba_middle_17m
        |   |   |   |-- config.py
        |   |   |   `-- run.sh
        |   |   |-- videomamba_middle_17m_unmasked
        |   |   |   |-- config.py
        |   |   |   `-- run.sh
        |   |   |-- videomamba_middle_25m
        |   |   |   |-- config.py
        |   |   |   `-- run.sh
        |   |   |-- videomamba_middle_25m_unmasked
        |   |   |   |-- config.py
        |   |   |   `-- run.sh
        |   |   |-- videomamba_middle_5m
        |   |   |   |-- config.py
        |   |   |   `-- run.sh
        |   |   `-- videomamba_middle_5m_unmasked
        |   |       |-- config.py
        |   |       `-- run.sh
        |   |-- exp_zs
        |   |   |-- anet
        |   |   |   |-- config.py
        |   |   |   `-- run.sh
        |   |   |-- didemo
        |   |   |   |-- config.py
        |   |   |   `-- run.sh
        |   |   |-- lsmdc
        |   |   |   |-- config.py
        |   |   |   `-- run.sh
        |   |   |-- msrvtt
        |   |   |   |-- config.py
        |   |   |   `-- run.sh
        |   |   `-- msvd
        |   |       |-- config.py
        |   |       `-- run.sh
        |   |-- models
        |   |   |-- __init__.py
        |   |   |-- backbones
        |   |   |   |-- __init__.py
        |   |   |   |-- bert
        |   |   |   |   |-- __init__.py
        |   |   |   |   |-- builder.py
        |   |   |   |   |-- tokenization_bert.py
        |   |   |   |   |-- tokenization_bert2.py
        |   |   |   |   `-- xbert.py
        |   |   |   |-- clip
        |   |   |   |   |-- bpe_simple_vocab_16e6.txt.gz
        |   |   |   |   |-- clip_text.py
        |   |   |   |   |-- tokenizer.py
        |   |   |   |   `-- transformer.py
        |   |   |   |-- videomamba
        |   |   |   |   |-- __init__.py
        |   |   |   |   |-- clip.py
        |   |   |   |   `-- videomamba.py
        |   |   |   `-- vit
        |   |   |       |-- __init__.py
        |   |   |       |-- clip.py
        |   |   |       `-- vit.py
        |   |   |-- criterions.py
        |   |   |-- mask.py
        |   |   |-- umt.py
        |   |   |-- umt_qa.py
        |   |   |-- umt_videomamba.py
        |   |   `-- utils.py
        |   |-- tasks
        |   |   |-- pretrain.py
        |   |   |-- retrieval.py
        |   |   |-- retrieval_mc.py
        |   |   |-- retrieval_utils.py
        |   |   |-- shared_utils.py
        |   |   |-- vqa.py
        |   |   `-- vqa_utils.py
        |   |-- torchrun.sh
        |   `-- utils
        |       |-- basic_utils.py
        |       |-- config.py
        |       |-- config_utils.py
        |       |-- distributed.py
        |       |-- easydict.py
        |       |-- logger.py
        |       |-- optimizer.py
        |       `-- scheduler.py
        `-- video_sm                                 # renyu: 单模态视频任务VideoMamba模型
            |-- DATASET.md
            |-- MODEL_ZOO.md
            |-- README.md
            |-- datasets
            |   |-- __init__.py
            |   |-- build.py
            |   |-- kinetics.py
            |   |-- kinetics_sparse.py
            |   |-- lvu.py
            |   |-- mae.py
            |   |-- masking_generator.py
            |   |-- mixup.py
            |   |-- rand_augment.py
            |   |-- random_erasing.py
            |   |-- ssv2.py
            |   |-- transforms.py
            |   |-- video_transforms.py
            |   `-- volume_transforms.py
            |-- engines
            |   |-- __init__.py
            |   |-- engine_for_finetuning.py
            |   |-- engine_for_finetuning_regression.py
            |   |-- engine_for_pretraining.py
            |   |-- engine_for_pretraining_umt.py
            |   `-- engine_for_pretraining_videomamba.py
            |-- exp
            |   |-- breakfast
            |   |   |-- videomamba_middle
            |   |   |   |-- run_f32x224.sh
            |   |   |   `-- run_f64x224.sh
            |   |   |-- videomamba_middle_mask
            |   |   |   |-- run_f32x224.sh
            |   |   |   `-- run_f64x224.sh
            |   |   |-- videomamba_small
            |   |   |   |-- run_f32x224.sh
            |   |   |   `-- run_f64x224.sh
            |   |   `-- videomamba_tiny
            |   |       |-- run_f32x224.sh
            |   |       `-- run_f64x224.sh
            |   |-- coin
            |   |   |-- videomamba_middle
            |   |   |   |-- run_f32x224.sh
            |   |   |   `-- run_f64x224.sh
            |   |   |-- videomamba_middle_mask
            |   |   |   |-- run_f32x224.sh
            |   |   |   `-- run_f64x224.sh
            |   |   |-- videomamba_small
            |   |   |   |-- run_f32x224.sh
            |   |   |   `-- run_f64x224.sh
            |   |   `-- videomamba_tiny
            |   |       |-- run_f32x224.sh
            |   |       `-- run_f64x224.sh
            |   |-- k400
            |   |   |-- videomamba_middle
            |   |   |   |-- run_f16x224.sh
            |   |   |   |-- run_f32x224.sh
            |   |   |   |-- run_f64x224.sh
            |   |   |   |-- run_f64x224to384.sh
            |   |   |   `-- run_f8x224.sh
            |   |   |-- videomamba_middle_mask
            |   |   |   |-- run_f16x224.sh
            |   |   |   |-- run_f32x224.sh
            |   |   |   |-- run_f64x224.sh
            |   |   |   |-- run_f64x224to384.sh
            |   |   |   |-- run_f8x224.sh
            |   |   |   `-- run_mask_pretrain.sh
            |   |   |-- videomamba_small
            |   |   |   |-- run_f16x224.sh
            |   |   |   |-- run_f32x224.sh
            |   |   |   |-- run_f64x224.sh
            |   |   |   |-- run_f64x224to384.sh
            |   |   |   `-- run_f8x224.sh
            |   |   `-- videomamba_tiny
            |   |       |-- run_f16x224.sh
            |   |       |-- run_f32x224.sh
            |   |       |-- run_f64x224.sh
            |   |       |-- run_f64x224to384.sh
            |   |       `-- run_f8x224.sh
            |   |-- lvu
            |   |   |-- run_class.sh
            |   |   |-- run_class_trim.sh
            |   |   |-- run_regression.sh
            |   |   `-- run_regression_trim.sh
            |   `-- ssv2
            |       |-- videomamba_middle
            |       |   |-- run_f16x224.sh
            |       |   |-- run_f16x224to288.sh
            |       |   `-- run_f8x224.sh
            |       |-- videomamba_middle_mask
            |       |   |-- run_f16x224.sh
            |       |   |-- run_f16x224to288.sh
            |       |   |-- run_f8x224.sh
            |       |   `-- run_mask_pretrain.sh
            |       |-- videomamba_small
            |       |   |-- run_f16x224.sh
            |       |   |-- run_f16x224to288.sh
            |       |   `-- run_f8x224.sh
            |       `-- videomamba_tiny
            |           |-- run_f16x224.sh
            |           |-- run_f16x224to288.sh
            |           `-- run_f8x224.sh
            |-- functional.py
            |-- models
            |   |-- __init__.py
            |   |-- clip.py
            |   |-- deit.py
            |   |-- extract_clip
            |   |   `-- extract.ipynb
            |   |-- modeling_finetune.py
            |   |-- modeling_pretrain.py
            |   |-- modeling_pretrain_umt.py
            |   |-- speed_test.py
            |   |-- videomamba.py                  # renyu: 一般的VideoMamba模型代码（在ImageNet上预训练）
            |   `-- videomamba_pretrain.py         # renyu: Masked预训练VideoMamba模型代码
            |-- optim_factory.py
            |-- run_class_finetuning.py            # renyu: 分类问题微调main函数
            |-- run_mae_pretraining.py             # renyu: 做Masked Autoencoder预训练？的main函数，实际未使用
            |-- run_regression_finetuning.py       # renyu: 回归问题微调main函数
            |-- run_umt_pretraining.py             # renyu: 做Unmasked teacher预训练？的main函数
            |-- run_videomamba_pretraining.py      # renyu: 应该是做Masked预训练的main函数
            `-- utils.py
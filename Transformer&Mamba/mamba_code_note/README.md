renyu的mamba源码阅读笔记，这里仅选了完整代码中一些核心的部分做注释，其他部分代码未放上来

参考Mamba整体的代码目录整理如下：

    .
    |-- 3rdparty                                        # renyu: 第三方库目录
    |   `-- lm-evaluation-harness                           # renyu: 用于语言模型性能评估的lm_eval库
    |-- AUTHORS
    |-- LICENSE
    |-- README.md                                       # renyu: Mamba项目说明文档
    |-- assets                                          # renyu: 资源目录，放一些文档图像
    |   `-- selection.png                                   # renyu: Selective SSM框图
    |-- benchmarks                                      # renyu: 测试代码目录，但仅提供了一个测试
    |   `-- benchmark_generation_mamba_simple.py            # renyu: 测试文本生成，除了Mamba还可以加载其他模型，会打印生成耗时
    |-- causal-conv1d                            # renyu: Mamba作者Tri Dao写的一个CUDA 1D Causal depthwise卷积的库，Mmaba中用到了
    |   |-- AUTHORS
    |   |-- LICENSE
    |   |-- README.md
    |   |-- causal_conv1d
    |   |   |-- __init__.py
    |   |   `-- causal_conv1d_interface.py       # renyu: causal_conv1d对底层CUDAExtension封装好的python API，装了pytorch可以直接调用
    |   |-- csrc                                 # renyu: causal_conv1d底层C++&CUDA源码，封装成一个CUDAExtension: casual_conv1d_cuda
    |   |   |-- causal_conv1d.cpp                # renyu: causal_conv1d主流程的C++代码，其中具体一些步骤调用了CUDA代码
    |   |   |-- causal_conv1d.h                      # renyu: 定义了causal_conv1d用到的struct类型
    |   |   |-- causal_conv1d_bwd.cu                 # renyu: causal_conv1d backward步骤CUDA代码
    |   |   |-- causal_conv1d_common.h               # renyu: TODO：格式转换+SumOp,Allreduce方法？
    |   |   |-- causal_conv1d_fwd.cu                 # renyu: causal_conv1d foward步骤CUDA代码
    |   |   |-- causal_conv1d_update.cu              # renyu: causal_conv1d update步骤CUDA代码
    |   |   `-- static_switch.h                      # renyu: 写了个bool swtich的宏CUDA代码中用了
    |   |-- setup.py                             # renyu: causal_conv1d包安装脚本
    |   `-- tests
    |       `-- test_causal_conv1d.py            # renyu: causal_conv1d单元测试代码
    |-- csrc                                            # renyu: Mamba S6结构底层selective_scan核心方法的C++&CUDA源码，封装成一个CUDAExtension: selective_scan_cuda
    |   `-- selective_scan
    |       |-- reverse_scan.cuh
    |       |-- selective_scan.cpp                      # renyu: selective_scan主流程的C++代码，其中具体步骤fwd和bwd调用了CUDA代码（还区分了不同精度）
    |       |-- selective_scan.h
    |       |-- selective_scan_bwd_bf16_complex.cu
    |       |-- selective_scan_bwd_bf16_real.cu
    |       |-- selective_scan_bwd_fp16_complex.cu
    |       |-- selective_scan_bwd_fp16_real.cu
    |       |-- selective_scan_bwd_fp32_complex.cu
    |       |-- selective_scan_bwd_fp32_real.cu
    |       |-- selective_scan_bwd_kernel.cuh
    |       |-- selective_scan_common.h
    |       |-- selective_scan_fwd_bf16.cu
    |       |-- selective_scan_fwd_fp16.cu
    |       |-- selective_scan_fwd_fp32.cu
    |       |-- selective_scan_fwd_kernel.cuh
    |       |-- static_switch.h
    |       `-- uninitialized_copy.cuh
    |-- evals                                          # renyu: 模型评估代码目录
    |   `-- lm_harness_eval.py                             # renyu: 对模型做zero-shot评估的代码，论文中的测试就是用的这个，用的lm_eval库
    |-- mamba_ssm                                      # renyu: Mamba模型主体代码，包括S6运算->Mamba Block->Mamba Model三个层次，分在三个目录
    |   |-- __init__.py
    |   |-- models                                     # renyu: Mamba Model代码目录
    |   |   |-- __init__.py
    |   |   |-- config_mamba.py
    |   |   `-- mixer_seq_simple.py                        # renyu: 使用Mamba Block堆叠成Mamba Model的示例代码，写了一个无输出层的基础模型、一个加Language Model Head输出的大语言模型
    |   |-- modules                                    # renyu: Mamba Block代码目录
    |   |   |-- __init__.py
    |   |   `-- mamba_simple.py                            # renyu: 基于S6运算代码封装了Mamba类，然后又加入残差、归一化等处理封装成Block类（即Mamba Block）
    |   |-- ops                                        # renyu: Mamba底层S6运算代码目录
    |   |   |-- __init__.py
    |   |   |-- selective_scan_interface.py                # renyu: 实现S6结构运算的核心代码，实际是对底层CUDAExtension的python API封装
    |   |   `-- triton                                     # renyu: 这里是因为归一化层和一个S6运算的代码是基于triton库实现的，所以加了个目录
    |   |       |-- __init__.py
    |   |       |-- layernorm.py                           # renyu: 归一化层代码
    |   |       `-- selective_state_update.py              # renyu: TODO: 这个代码意义存疑，推测是递归方式做S6运算，可能是仅参考实现或推理使用
    |   `-- utils                                      # renyu: Mamba用到的一些其他组件代码
    |       |-- __init__.py
    |       |-- generation.py                              # renyu: TODO: 这个代码意义存疑，定义了GenerationMixin类在Mamba LLM模型中作为基类
    |       `-- hf.py                                      # renyu: 从hugging face加载预训练模型、配置参数的代码
    |-- setup.py                                       # renyu: 包安装脚本，安装问题可以在这里对一下依赖
    `-- tests                                          # renyu: 单元测试代码，测了底层核心的方法
        `-- ops
            |-- test_selective_scan.py                     # renyu: selective_scan单元测试
            `-- triton
                `-- test_selective_state_update.py         # renyu: 递归S6运算单元测试？

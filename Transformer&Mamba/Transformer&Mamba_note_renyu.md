# Transformer
## 概述
2017年提出的新的神经网络架构，核心思想是注意力机制，从而使得网络更能够利用长文本的语义关联，即更好的理解上下文，非常适合复杂的NLP任务。  
除此之外Transformer可以并行化计算，就提升了训练和推理的效率，使得基于Transformer做参数量庞大的LLM成为可能。  
## 经典论文
### 《Attention is All You Need》
2017 NIPS经典论文  
提出了Transformer模型，仅使用注意力机制，没有使用经典的RNN/CNN模型要用的递归/卷积。
#### 网络结构分析
主要分为编码器encoder和解码器decoder两部分，当然细分一点还可以加上输入的预处理部分和输出部分。  
##### 输入
直接输入的是一段文本，但是机器学习模型不能直接处理一段文本，会先做tokenize分词，英文就是变成一个一个单词，中文就是变成一个一个字，称为一个token。然后再做embedding即词向量化，输入一个token输出一个高维的向量，例如512维的向量，占512个字节。这里输入token到输出词向量之间的映射关系可以根据一些已经预训练好的词表，例如谷歌word2vec里提供了一些词表，也可以自己做embedding层自己训练。Transformer文章里就是自己训练的。  
由于Transformer不做递归或者卷积，就完全没有位置信息，分词后所有词都是一样输入的，失去了语序。所以还另外引入了positional encoding位置编码，做法是根据当前词位置生成一个位置向量，维数和词向量一致例如也是512维，然后和词向量直接加在一起就行，最后输入还是512维。（TODO：具体研究下为什么这样就有效？）  
##### Encoder
包含N个重复的block，文章中用的是6个。  
一个block的结构为4个部分：Multi-Head Attention多头注意力机制结构->Add & Norm结构->Feed Forward结构->Add & Norm结构  
* Multi-Head Attention结构  
非常重要，是Transformer的核心。理解多头注意力机制结构，先看更基础的Scaled Dot-Product Attention缩放点积注意力结构，可以计算出词向量两两之间的注意力权重（？），大概是可以获得词之间的关联关系，从而学习到长距离的语义依赖。输入有三个，查询向量Q，键向量K，值向量V。（TODO：这里具体的运算没有看的明白，QKV对应是什么东西……）大致看了下，首先计算Q和K，得到Q和K中向量两两之间的注意力权重/依赖关系（？），然后要做归一化。最后在用这个权重去乘以V。多头注意力机制应该就是搞多个Scaled Dot-Product Attention结构再去加权求和，效果会更好。（TODO：为什么？）  
* Add & Norm结构  
add应该是加了残差可以跳层，norm归一化用的是NLP中常用的Layer Norm层  
* Feed Forward结构  
说结构上很简单是两个全连接层，中间是ReLU激活函数。（TODO：为什么这么设计？）看有的文章说多头注意力只是获得词与词之间的关系，但这里才是实际做非线性变换学习其中信息的，有点抽象不是很理解。  
##### Decoder
结构上基本和Encoder一致，也是N个block，区别在于每个block前多了一个Masked Multi-Head Attention结构+Add & Norm结构，就是2+4=6个部分了。  
Mask的思想说是很简单，就是在解码预测的时候，不让模型预知所有信息，都是从头开始根据已知的文本一点一点推断生成。（TODO：也是有点抽象要研究下）


### 《BERT: Pre-training of Deep Bidirectional Transformer for Language Understanding》
2018 SOTA的NLP模型，1.1亿参数 1.56TB训练数据，在当时已经是最强的，使得谷歌翻译等应用直接进步了一大截。  
网络结构是Transformer的变体，只使用编码器结构去除了解码器结构，并且使用双向编码器结构可以学习反向信息。  
BERT是基于Transformer预训练+微调的模型，也就是不关新具体任务直接大量数据对模型进行预训练（具体到BERT这里就是无监督数据，大量的文本），然后针对具体要做的有监督任务进行微调，例如机器翻译等。做的预训练效果很好，大概是做了两个阶段的事情，第一阶段Masked Language Modeling掩码语言建模是屏蔽文本中的一些词然后让模型去预测；第二阶段Next Sentence Prediction是输入两个句子然后判断这两个句子是否上下文相关。后续GPT3、4也都是基于Transformer预训练+微调的模型。  

## 代码
谷歌官方是T2T团队开源的代码 https://github.com/tensorflow/tensor2tensor  
有博主说代码比较难读，后面其他人的实现也不少，建议看哈佛NLP团队写的pytorch注释版本https://github.com/harvardnlp/annotated-transformer




# Mamba
## 概述
Transformer模型很好，但是存在的缺陷是自注意力机制的计算量，随着上下文长度的增加呈平方级增长  
Mamba网络结构的核心是Selective State Space Model 选择性状态空间模型  
Mamba的计算量随上下文长度是线性增长的  
通用序列模型，所以文本、音频、基因组等都可以用  
作者只有两位，CMU的Albert Gu和普林斯顿的Tri Dao  
主要创新点是选择性SSM架构，基于Albert Gu之前做的S4（Structured State Space Model）架构泛化而来，所以学习的话推荐去读the annotated s4文章（同时推荐the annotated transformer），对S4的细节做了很多注释解读（大致看了感觉有很细致的数学推导，对于我而言还是有一点难读，可能需要找时间硬啃下）  
提出S4就是因为当时有长上下文的任务Long Range Arena和Path X等，很多模型都解决不了

## 网络结构
网络结构不算是很难理解，就是堆叠多层mamba block，mamba block中间就有Selective SSM结构。  
### Selective State Space Model
理解Selective SSM结构是理解mamba网络的核心，之前提出了很多SSM结构在尝试解决Transformer对于长序列的计算量问题，但是最终预测准确性都比Transformer差，相当于牺牲准确性省时间和内存，但没有意义了。作者分析了其中的原因是SSM没有基于内容推理的能力，所以做了改进，实现了接近Transformer的性能并降低了计算量和内存开销。  
SSM的原理和RNN类似（TODO：学习研究下）  
作者提出的Selective SSM创新点在于选择，作者认为选择机制是一种信息压缩的方法，在序列循环的过程中要筛选出有用的信息。  
TODO：论文有个结构图要贴过来  
整个过程还是比较复杂的，详细的过程需要对照代码去看（TODO）。  
这个计算的好处是很好并行计算。  
### Mamba Block
TODO：论文有个结构图要贴过来  
TODO：过程没看太明白，也需要对照代码看  

## 代码
https://github.com/state-spaces/mamba  
主要计算代码在mamba_ssm目录下  
mamba_ssm/ops/selective_scan_interface.py 是Selective SSM结构的代码  
mamba_ssm/modules/mamba_simple.py 是mamba block的代码  
mamba_ssm/models/mixer_seq_simple.py 是mamba模型的代码

看到有另外一个简化版本的mamba代码推荐学习，可读性更好  
https://github.com/alxndrTL/mamba.py  

# Mamba与视频
## VMamba: Visual State Space Model
TODO
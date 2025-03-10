# Transformer  
## 概述  
2017年提出的新的神经网络架构，核心思想是注意力机制，从而使得网络更能够利用长文本的语义关联，即更好的理解上下文，非常适合复杂的NLP任务。  
除此之外Transformer可以并行化计算，就提升了训练和推理的效率，使得基于Transformer做参数量庞大的LLM成为可能。  
## 经典论文  
首先要知道非常有名的Transformer中用的是自注意力机制self attention，注意力机制并不是Transformer首次提出的，在此之前有两个重要的注意力机制给予了启发。  
### 《Neural Machine Translation by Jointly Learning to Align and Translate》  
2014年 Dzmitry Bahdanau提出了Bahdanau注意力机制。依然基于RNN。  
### 《Effective Approaches to Attention-based Neural Machine Translation》  
2015年Minh-Thang Luong提出了Luong注意力机制。基于LSTM。  
### 《Attention is All You Need》  
2017 NIPS经典论文  
提出了Transformer模型，仅使用注意力机制，没有使用经典的RNN/CNN模型要用的递归/卷积。  
#### 网络结构分析  
主要分为编码器encoder和解码器decoder两部分，当然细分一点还可以加上输入的预处理部分和输出部分。  
##### 输入（重点要理解格式）  
###### 1.文本序列-分词->token序列  
直接输入的是一段文本，但是机器学习模型不能直接处理一段文本，会先做tokenize分词，英文就是变成一个一个单词，中文就是变成一个一个字，称为一个token。  
###### 2.token-embedding向量化->高维词向量  
然后再做embedding即词向量化，输入一个token输出一个高维的向量，例如512维的向量，占512个字节。这里输入token到输出词向量之间的映射关系可以根据一些已经预训练好的词表，例如谷歌word2vec里提供了一些词表，也可以自己做embedding层自己训练。Transformer文章里就是自己训练的。  
###### 3.加入位置信息->词向量+位置编码向量  
由于Transformer不做递归或者卷积，就完全没有位置信息，分词后所有词都是一样输入的，失去了语序。所以还另外引入了positional encoding位置编码，做法是根据当前词位置生成一个位置编码向量，为了保持向量维数不变，位置编码向量维数和词向量一致例如也是512维，然后和词向量直接加在一起就行，最后输入还是512维。（TODO：具体研究下为什么这样就有效？）  
##### Encoder  
包含N个重复的block，文章中用的是6个。  
一个block的结构为4个部分：Multi-Head Attention多头注意力机制结构->Add & Norm结构->Feed Forward结构->Add & Norm结构  
太多了记不住，更简化一点理解也可以认为是两部分，一个自注意力层，一个前馈神经网络层。  
###### 1 Multi-Head Attention结构  
**非常重要，是Transformer的核心，认真理解！！！！！**  
这里的“多头”指的是同时搞了多个注意力结构去加权求和，获得更好的效果。为什么要多头？这里的原因还挺多讨论的，直观理解很简单，我做多次去加权平均肯定是更好的。一方面是可能多个注意力机制有可能发现序列中不同的依赖关系（这个有点抽象，我看到有讨论说大部分情况下结果其实是差不多的……）；另一方面是会有部分情况单个注意力机制过拟合出现问题，那这个时候多头就更健壮，不容易过拟合。再看下缺陷是引入了更多参数和计算复杂度，但反正可以并行计算只要算力&内存够不会影响时间开销，可以接受。  
**下面具体到一个注意力机制结构，看传说中的注意力机制是如何实现的！** 首先我个人觉得“注意力”这个名字起的好听但是不准确，容易让人学习的时候产生误解，再加上一些教程的引入喜欢说人会关注哪一部分是重点，让学习者误以为注意力和人的主观注意力有什么关联。我觉得更准确的说是上下文依赖关系/相似性这样更方便学习中理解。输入一个序列各个词之间能有啥注意力，说一个词注意另一个词真的很奇怪……也有人说这个翻译特别好的……  
实现自注意力机制的结构叫做Scaled Dot-Product Attention缩放点积注意力结构。这个东西设计的很神奇，我想不出来这么复杂的一通运算是怎么想出来的，不像是RNN、LSTM结构那样还有一些更直观的含义。它的效果是可以计算出词向量两两之间的注意力权重（我觉得就是依赖性权重，n个词向量就得到n\*n的矩阵，和协方差矩阵的感觉比较像），大概是可以获得词之间的关联关系，从而学习到长距离的语义依赖。  
这里的运算原理和意义是需要花一点功夫去理解的，还是比较抽象复杂。输入有三个，查询向量Q，键向量K，值向量V。（TODO：有很多文章去抽象分析，多学习下，理了好多遍还是不太对）大致看了下，QKV其实都是输入的序列X线性变换得到的，设输入序列长度为L即L个词向量，每个词向量是d维，则X是L\*d维矩阵。Q、K、V分别有三组权重矩阵$w_q,w_k,w_v$，都是d\*d维，然后X分别乘以这三个权重得到Q、K、V矩阵，所以乘完之后Q、K、V还是L\*d维。计算方法看起来一样，但是意义是不一样的，很多文章说Q矩阵理解为待查询的token，K理解为带匹配的键，V是K对应的结果，但是看了计算我觉得完全无法理解。  
计算过程：首先计算Q和K转置的乘积，矩阵乘矩阵转置的意义是得到Q和K中向量两两之间的注意力权重/依赖关系（或者说相似度矩阵）。然后要除以一个系数$\sqrt{d_k}$，这一步是控制方差方便梯度更新。再做softmax归一化，得到的算是一个权重矩阵。最后再用这个权重去乘以V，相当于加权求和，得到的是序列中所有词向量更新权重后的新表示（TODO：这里的输入输出结果我还是没有确认清楚，我认为最终注意力结构的输入L\*D的输入词向量序列矩阵，输出是L\*L的词向量间注意力权重矩阵。但是看一个图解的文章似乎不一样，还需要再研究下）。公式为$Softmax(\frac{QK^T}{\sqrt{d_k}})V$  
###### 2 Add & Norm结构  
add应该是加了残差可以跳层，norm归一化用的是NLP中常用的Layer Norm层  
###### 3 Feed Forward结构  
这里有一点概念辨析的问题，我听Feed Forward前馈这个词不是熟悉，但实际上这是现代神经网络最宽泛的统称，因为前馈指的就是从输入到输出的单向传递，所有“普通”的神经网络其实都是这样的。所以不用纠结这个名字，这里就是为了区分前面是创新性的多头注意力机制结构，后面就是旧的神经网络。  
理论上这里用各种神经网络结构都是可以的，Transformer这里结构很简单就是两个全连接层，中间是ReLU激活函数。至于为什么这么设计：看有的文章说多头注意力只是获得词与词之间的关系，但这里才是实际做非线性变换学习其中信息的，有点抽象不是很理解。  
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
  
## Transformer in CV  
这里不太关心具体的理论基础了，我就不从论文出发整理了，重点找一些带好的开源代码的模型，从模型出发整理  
找合适模型代码的方法和找论文不一样，论文好不一定代码好使，但是大家用的多的代码一定是好使的  
上Hugging Face查热门模型  
上Github查高star项目  
看魔改模型项目常用的baseline、Backbone  
#### 图像-Pytorch Timm库  
https://github.com/huggingface/pytorch-image-models  
Ross Wightman大神创建，集成了大量CV任务的模型  
  
#### 视频-SlowFast、PyTorchVideo库  
https://github.com/facebookresearch/SlowFast  
https://github.com/facebookresearch/pytorchvideo  
Facebook整的库，集成了一些经典的Video模型，主要是C2D、I3D、SlowFast、CSN、X3D等，有Facebook搞的MViT（Multiscale Vision Transformers）  
  
#### 图像-（20谷歌ViT CV任务Transformer奠基）《An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale》  
https://github.com/google-research/vision_transformer  
2020年将Transformer用在CV任务的ViT，Google团队做的。不算最早，但是实现了最简单的方式取得很好的效果，并且实验做的非常完善。成为了奠基论文。（之前尝试Transformer用于CV任务效果不好，其实有数据量不足的问题）  
最重要的一点就是输入的patch embedding。将224x224的输入图像，分为16x16的小patch，一张图片对应的序列长度就是(224x224)/(16x16)=196，一个patch Embedding之前的维度是16x16x3通道=768维。  
然后一个patch经过线性投影层就得到一个token/Embedding，这里我一开始不理解啥是线性投影层，其实看下代码实现就是不加ReLU激活函数的2D卷积，然后卷积核大小同Patch 16x16x3，步长也同Patch 16x16，这样就实现了一个Patch-卷积核对应位置点乘累加得到一个值，那想要任意长度的Embedding，只需要调整卷积核数量即可。原始的ViT中就是用了768个卷积核保持维数不变。  
既然用了卷积（线性投影）层，这里就有可以学习的weight和bias。所以和NLP任务中使用固定的预训练模型/word2vec等工具做Embedding不同，这也是由于CV任务中图像输入比文本输入更复杂，没法用固定的Embedding方式就得到准确的Representation，所以ViT中的Embedding层也是一起训练调参的。  
  
#### 图像-（21Facebook DeiT）Data-Efficient Image Transformers  
https://github.com/facebookresearch/deit  
高质量的Video Transformer代码  
DeiT主要做的事情就是原版ViT要的数据量很大，在小数据集上效果不好，所以引入了模型蒸馏和数据增强，能取得更好的效果  
  
#### 图像-（21微软 Swin Transformer）Swin Transformer: Hierarchical Vision Transformer using Shifted Windows  
https://github.com/microsoft/Swin-Transformer  
经典的Swin Transformer，是微软的官方实现，代码质量高  
  
#### 视频-（21Facebook TimeSFormer）Is Space-Time Attention All You Need for Video Understanding?  
https://github.com/facebookresearch/TimeSformer  
比ViViT还早一点点的一个视频Transformer实现，也是尝试了不同的方式来处理时间维度  
对比了  
1.仅空间  
2.普通时间空间一起，计算量很大  
3.先时间后空间（认为的最好方式）  
4.先局部后全局？  
5.轴方向切分，先时间后分别宽和高？  
没有细研究，看指标也不错，但好像相比ViViT比较少提，区别乍一看就是先做了时域的注意力？  
  
#### 视频-（21谷歌）ViViT: A Video Vision Transformer  
https://huggingface.co/docs/transformers/en/model_doc/vivit  
https://github.com/google-research/scenic/tree/main/scenic/projects/vivit  
代码说是基于谷歌自己推的JAX开发的，不是pytorch，好像比较少使用？  
2021年将Transformer扩展到视频任务的ViT，奠基论文  
讨论了输入2D图像加入时间维度成3D视频后应该如何使用Transformer处理  
使用Tubelet Embedding（就是3D小长方体作为一个patch），并讨论了4种对3D视频输入的注意力机制model看怎么提取时空特征比较好，分别是  
1.时间空间不分开处理直接原始token输入，简单但计算量大（我理解是不是普遍还是这样的简单方式，直接送进去）  
2.完全先空间再时间，相当于过两个Encoder，帧内提空间token过空间Encoder，然后得到的token再合在一起过时间Encoder  
3.局部先空间再时间，修改了Transformer Block的结构，每一个Block中先算空间注意力再算时间注意力  
4.局部分开空间时间，修改了Transformer Block的结构  
  
#### 视频-（21亚马逊 VidTr）VidTr: Video Transformer Without Convolutions  
https://github.com/amazon-science/gluonmm/tree/main/src/transformers/models/vidtr  
也是早期的一个Video Transformer模型，但是做的可能没那么好，只是部分文章里会提到，大部分会忽略这个  
  
#### 视频-（21 Video Swin Transformer）Video Swin Transformer  
https://github.com/SwinTransformer/Video-Swin-Transformer  
代码是基于mmaction2开发的，确实是很好的库，但依赖项会有点多  
FastVQA中用的SwinTransformer3D似乎不是这个  
  
#### 视频-（22南京大学 VideoMAE）VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training  
https://github.com/MCG-NJU/VideoMAE  
这个事情做的很牛，在视频模型上跑无监督学习  
原理也是类似NLP的无监督学习，对视频数据进行部分掩码，然后训练模型恢复掩码部分，效果非常好，是Video Transformer的突破性论文，但是在跑训练过程中也需要很大的算力  
后面23年还出了第二版  
类似的视频MAE也有一些其他研究，例如  
（22Facebook ST-MAE）  
Masked Autoencoders As Spatiotemporal Learners  
（23复旦 MVD）Masked video distillation: Rethinking masked feature modeling for self-supervised video representation learning  
后面可以了解下  
  
# 大语言模型&大视觉模型&多模态大模型  
## 大语言模型  
### 概述  
17年搞出Transformer之后，机器翻译等序列任务都搞出了很好的结果，可以视为一个更强大的神经网络模型，但更关键的潜力在于scalable，可以分布式训练，可能能做出很大size的模型。但问题是，越大的模型就需要越多的训练数据，哪里有这么多标签数据？  
没有那么多标签数据，但是有的是无标签数据啊。如果能引入无监督预训练，超大数据集+超大模型，可能就有意想不到的效果。这就是OpenAI Ilya的思路，越大的模型越牛逼，想办法往上堆就行  
还存在的一个问题是怎么去无监督预训练才有用？18年的时候出了两篇奠基论文直接确认了这个方向可以搞。一篇是OpenAI的GPT-1，一篇是谷歌的BERT。这两篇论文分别引入了不同的两种预训练方式，效果都非常好，模型直接就有了文本生成的能力。  
* 自回归模型  
这种名字听着有点唬人，就是同时间信号处理里的AR模型，当前值仅由历史值来预测这个概念。说白了就是引入的预训练方式为输入前面句子，预测下一个词。输入大量文本资料去预测下一个词，这样大语言模型就自然可以通过不断生成下一个词的方式来生成长文本了。  
* 自编码模型  
预训练方式就是掩码补全，mask掉句子里的部分词后预测恢复原本的句子。  
  
知道这个方向有搞头之后，后面就可以继续scale up了，后续有很多经典的早期模型：  
GPT-1 -> GPT-2 -> GPT-3  
BERT -> RoBERTa/DistilBERT  
T5  
对比一下会发现自回归模型文本生成效果非常好，所以直接延伸出后面各种chatGPT等工具。自编码模型跑各种NLP任务效果好，直观看起来没有那么流行了。  
### 经典论文  
#### （18.6.11OpenAI GPT-1）Improving Language Understanding by Generative Pre-Training  
  
#### （18.10.11谷歌）BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding  
  
## 大视觉模型 LVM  
### 概述  
很直接的思路就是CV任务上能不能复刻LLM的成功  
  
## 多模态大模型 MLLM  
### 经典论文  
#### （21.2.26OpenAI CLIP）Learning Transferable Visual Models From Natural Language Supervision  
OpenAI从LLM做过来很直接的想法就是搞超大图像-文本数据集预训练，有几个核心的问题。  
* 1. 什么样的文本数据？  
标准的图像分类数据集应该是有多个类别标签，也是文本信息。但CLIP中决定用自然语言。一方面是因为自然语言的描述好从互联网上搞，分类标签太标准了只能人工标记；另一方面是自然语言有更多的语义信息，潜力更大。  
* 2. 超大规模训练数据哪里来？  
常用的几个图像标注数据集对于大模型训练而言太小了，超大规模至少上亿级别。所以作者自建了训练数据为4亿的图像-文本对数据集WIT（WebImageText）。  
* 3. 怎么做预训练？  
图像-文本配对问题，直接能想到的做法就是输入图像，给出预测文本，但是这个事情实际不好做，给定一个图有无数种描述方式，很难精确给出文本预测。  
退而求其次改成对比学习来匹配图像-文本对发现效果还可以，目标是提升匹配对嵌入的余弦相似度，降低非匹配对嵌入的余弦相似度。（定义了特别的损失函数对称交叉熵）  
我开始不知道如此大的数据量怎么做对比学习，以为是4亿样本中随便选，后面看应该是一个batch中打乱去匹配，用很大batch如256这样还是可以。  
* 4. 模型结构？  
网络结构是图像编码器将图像转为向量（最后选了ResNet），文本编码器是Transformer。分类器是一层逻辑回归？  
  
不是很理解这怎么做的，还是得看代码。  
有一个关键点是我一开始没理解的，把CLIP当做一种LLM了，以为输出是文本数据。其实CLIP就是训练一个图像encoder+一个文本encoder，通过对比学习不断输入图像-文本对，从而获得一对匹配的图像&文本encoder，使得生成的向量可以通过计算相似性判断图像&文本的匹配程度。所以后续就可以做一些图像-文本匹配的任务，例如图像检索等。  
  
# Mamba  
## 简介  
非常有影响力的新LLM架构，是Transformer的有力竞争对手。作为通用序列模型，可以用于文本、音频、基因组等各种任务。  
来自于论文 Mamba: Linear-Time Sequence Modeling with Selective State Spaces。23.12.1新挂到arXiv上，https://arxiv.org/abs/2312.00752  
作者只有两位，CMU的Albert Gu和普林斯顿的Tri Dao，两人应该是斯坦福的同学，刚毕业不久的新AP，曾经都是同一个导师Christopher Ré的学生  
mamba论文曾经投递ICLR 2024但是8/8/6/3被拒了，引起了一些争议，审阅记录在https://openreview.net/forum?id=AL1fq05o7H  
目前mamba架构已经被用在一些任务上，证明取得了非常好的效果  
## 动机——现有序列模型在长序列问题上的局限性  
首先介绍下序列模型的概念，就是输入为一维序列，输出也是一维序列，相当于做了个序列到序列映射的数学模型。（不过也有看定义是只要输入输出一侧是序列就可以）最常用的序列数据就是文本数据，所以序列模型也是NLP中的核心数学模型。  
已经有很多经典的序列模型，包括Neural ODEs、RNN、CNN、Transformer，其中Transformer的效果太好了，在各项NLP任务中已经取得了非常好的效果，也因此造就了LLM的发展。  
Transformer模型很好，但是存在的缺陷是自注意力机制的计算量，需要计算输入序列两两词向量之间的注意力依赖（？），所以计算量随着上下文长度的增加呈平方级增长，即时间复杂度$O(L^2)$。这就造成了Transformer的一大缺陷，无法处理很长的上下文信息，一般可能只保留固定大小的窗口（例如早期chatgpt 2000个Token，后面支持到4k 8k，声称可以最大128k等），输入过大则旧的信息就无法计算注意力被抛弃了。  
2000个Token对于一些不算很长的文本数据而言能记录一些上下文还是可以的，但是如果有更长更复杂的序列数据，例如高分辨率图像、视频、音频（每秒16000+采样点）就不够用了。所以一些长上下文的任务Long Range Arena和Path X等，当前时刻的结果理论上要受到所有历史信息的影响，也就是要“无限长”的记忆，一直没有好的模型去解决。  
因此一直有很多针对降低Transformer计算复杂度的研究，例如linear attention、gated convolution、recurrent model以及一些SSM (state space model)，研究者希望用这些计算复杂度较低结构替代Transformer，但是就整体结果来看准确率并不好，甚至远落后于注意力机制。  
### 序列模型的核心问题——处理远程依赖关系LRD  
远程依赖关系概念人很好理解，就是历史数据和现在数据有某种关联，但是让机器做起来很难。  
更抽象的角度来说，是如何实现记忆。内存有限算力有限，不可能把所有的历史数据都原封不动存起来，那如何“记住”历史数据？  
这个问题太抽象了，不同的序列模型族（model family）给出了完全不同的理解方式和解决思路。  
理解Mamba，重点要理解RNN、Transformer、SSM实现“记忆”的方式。其次还有CNN和Neural ODEs。  
RNN——记忆是特征。不断更新（历史特征+当前输入）合起来提取的特征，提取到的hidden state输出中就包含了所有历史的特征  
Transformer——记忆是信息之间的关系（也是种特征）。计算窗口内所有token互相之间的关联，不存在遗忘。  
SSM——记忆是拟合。用有限的记忆不断更新对于所有历史数据的最佳拟合  
## 研究基础&背景知识  
直接看懂Mamba其实还是有一定知识要求，这里网前理一下Mamba的前身研究，按顺序来看有利于理解。其实就是沿着Albert Gu大神的研究思路看看。  
作者不是凭空就提出了Mamba网络的，Albert Gu之前一直就是做SSM的研究尝试解决长序列的“记忆”问题，曾经提出过提出S4（Structured State Space Model）结构，（来自ICLR 2022 7篇outstanding parper HM之一的Efficiently Modeling Long Sequences with Structured State Space，https://arxiv.org/abs/2111.00396）在SSM中是比较厉害的，但是仍然比不上注意力。  
这个结构也就是mamba核心结构的前身，要学习mamba模型的原理就需要研究明白S4的原理。  
学习的话推荐去读the annotated s4文章（同时推荐the annotated transformer），对S4的细节做了很多注释解读（大致看了感觉有很细致的数学推导，对于我而言还是有一点难读，可能需要找时间硬啃下）  
而S4的前身是作者Alber Gu和Tri Dao一起提出的Hippo算子，（来自NeurIPS 2020 Spotlight论文HiPPO: Recurrent Memory with Optimal Polynomial Projections，https://arxiv.org/abs/2008.07669）  
后面Albert Gu和Tri Dao在2021年又进一步分析了基于Hippo算子，结合RNN、CNN、微分方程的优势如何实现最好的序列模型，结论就是个SSM。（来自NeurIPS 2021 Combining Recurrent, Convolutional, and Continuous-time Models with Linear State Space Layers）  
  
作者分析了很多SSM结构，认为关键的问题是它们无法进行基于内容的推理。  
而，并且能做到比较好的效果。  
而mamba模型主要创新点是Selective State Space Model选择性状态空间模型，即Selective SSM结构，就基于Albert Gu之前做的S4结构修改而来，作者把其记为S6。  
  
下面按照顺序补充一些学习Mamba的背景知识循序渐进。  
### State Space Model 状态空间模型  
Albert Gu推荐了https://probml.github.io/ssm-book/root.html电子书有点SSM简介  
这个是很经典的模型了，并不是新的技术。是连续时间下表示和分析动态系统的经典数学模型。挺多领域都会用到，例如搞控制的同学会比较熟悉，在现代控制理论课中一上来就会学习状态空间模型，深度学习这边属于借鉴，因为发现很多理论搞来搞去比较像是状态空间模型。  
具体可以用两个公式表示：  
$x'(t)=Ax(t)+Bu(t)$  
$y(t)=Cx(t)+Du(t)$  
这里考虑的是连续系统，$x(t)$代表系统状态（可能是多维向量），$x'(t)$导数代表系统状态变化量，$u(t)$是系统输入的信号，$y(t)$是系统的输出。关于参数D，似乎是和噪声有关，做深度学习中一般直接设D=0忽略掉（说是可以看做skip connection简化计算）  
TODO：我一直没想明白这个公式怎么来的。但是这样建模的意义就是：只要知道其中的系数A B C D（相当于一个最小的信息）就可以确定这个系统，已知输入u和当前状态x，就可以得到系统输出y和当前变化量x'，不需要去记录大量的历史数据。  
高层次一点理解，SSM实现记忆的方式，是通过统计学的方式，为系统建模，拟合一个系统的概率分布，从而只要少量的模型参数就可以大致记下来这个系统的所有历史信息，而不需要确切地去记录系统的所有历史状态数据。理解不了具体公式先大致理解这个思想。  
#### SSM序列模型问题一 连续模型离散化  
SSM最初是用于解决一些物理系统、控制系统的跟踪和预测问题，所以系统的状态都是一个连续时间的函数。但是用于序列模型，我们要处理的是一个离散的序列输入。直观而言离散序列不过是对于连续输入信号的采样，做是可以做的，但是也需要一些处理没有那么简单。  
状态空间方程经典的离散化方法有欧拉方法、双线性变换、零阶保持法等。作者在S4的处理中用的是双线性变换，然后会把经典的SSM模型方程化为离散的形式（省略系数D）。  
$x_k=\overline{A}x_{k-1}+\overline{B}u_k$  
$y_k=\overline{C}x_k$  
#### SSM序列模型问题二 递归计算化卷积计算  
化为了离散形式，可以看出来形式其实有点像是RNN的hidden state计算（也有mamba这种SSM就是去掉非线性的RNN的说法），那我们计算的时候要不断递归了，参考RNN计算就可以知道递归计算无法并行，效率会非常低，这是没法做大模型的。但是我们其实可以通过数学变换把这种递归计算化为卷积计算，使得其可以并行。  
TODO：具体的数学推导和结论也是没有能看懂需要再研究下细节。大致是引入了一个SSM的卷积核$\overline{K}$  
### Hippo——实现实时增量长期记忆的数学工具  
2020年的时候作者提出的一个算子，将任意函数投影到正交多项式空间上，从而实现了输入一个函数随着时间的推移快速增量更新，获得最优的多项式近似值。简单来说就是可以“实时”（online）生成一个时间函数到当前时间t的最优多项式拟合结果。  
多项式展开这个大家都知道了，例如泰勒展开、勒让德展开，说明了可以用多项式来拟合任意函数。假设我们值使用N次以内的多项式，那拟合能力是有限的，就需要实现能力范围内的最优多项式近似。（如何评价近似的质量，这里用到了函数内积啥的来评估函数相似性，理论上函数内积越大越相似，细节可以研究下）假设要记的东西就是一个连续时间函数（用连续函数比较好理解一些，实际应用都是离散的时间序列也没问题，论文里给出了Hippo算子离散形式的推导），想一想如果随着时间推移，我们能一直仅记录N个多项式的系数（有限的记忆空间），但是保持着最优近似来拟合从t_0时间到当前t时间的函数，那就是实现了无限长的“记忆”，非常高明的想法。抽象的记忆问题，转换为实时函数拟合的问题。  
HIppo应该是做的勒让德多项式分解，可以实时获得勒让德级数各项的系数。  
甚至作者还分析了不用多项式只用一阶线性拟合的情况下，式子和门控RNN是一样的，所以GRU这些方法相当于仅投影到1阶的Hippo算子。这样说起来Hippo肯定更厉害。  
集成到RNN中直接实现了在当时长依赖任务上的最优结果。  
TODO：但是Hippo的数学推导我看起来很吃力……  
#### Linear State Space Layer (LSSL)  
2021年作者基于Hippo开始思考如何结合递归RNN、卷积CNN、微分方程Neural ODEs方法的优势来做序列模型。提出了LSSL。  
递归可以做有状态的推理，卷积并行训练快，微分方程可以时间尺度自适应（？），有没有办法全都要？LSSL就是的。这个时候作者也在思考如何看待LSSL，认为它其实是一个离散的SSM。  
在Hippo的文章中其实还并没有将其用到SSM中的想法，还只是一个基础的数学工具，验证的时候也是放到RNN里。这篇文章里Hippo+SSM作者就证明了其对于长期记忆的有效性，为后续的S4 S6奠定了研究方向。  
我理解这里的结论是，在上面的SSM方程中，如果用特殊的Hippo矩阵作为SSM的系数矩阵A，就可以使得状态x记住输入u的历史信息，从而实现长期依赖的记忆。  
### Structured State Space Model (S4)  
Hippo算子理论上解决了使得SSM能处理长距离依赖的记忆问题。但是，实际是不太好用的，因为计算量还是很大的，假设输入序列长度为L，多项式阶数为N（TODO：这里N和L的意义我还不太确认……），那计算时间复杂度是$O(N^2L)$，空间复杂度是O(NL)，说作者用的N=256，这么大在计算中已经开销非常大了。  
所以作者做了一系列很牛的数学操作，来降低使用Hippo算子的SSM的计算复杂度，这就有了S4。TODO：整体的数学推导实在是有点复杂，作者真的很强。大致的内容是矩阵对角化可以减低计算复杂度，但是直接把Hippo矩阵对角化失败，会数值溢出。所以作者把Hippo矩阵转换为了正规矩阵（可以分解为对角矩阵）和低秩矩阵的和，然后就可以引入更快速一些的计算方法，最后把计算复杂度降低到了$O(N+L)$。  
然后作者分析了S4的整体的推理、训练的计算复杂度，并且和RNN、卷积、自注意力机制做了对比，严格证明了S4的计算复杂度是最低的。  
作者又通过消融实验证明了用Hippo矩阵的关键性。  
  
## 网络结构   
### Selective State Space Model  
理解Selective SSM结构是理解mamba网络的核心，之前提出了很多结构包括SSM在内在尝试解决Transformer对于长序列的计算量问题，但是最终预测准确性都比Transformer差，相当于牺牲准确性省时间和内存，但过差的准确性就失去了应用的意义。作者很抽象地分析了其中的原因是SSM没有基于内容推理的能力（有点高水平），所以针对这一点引入选择机制做了改进，从而实现了接近Transformer的性能并降低了计算量和内存开销。Mamba模型计算量随上下文长度是线性增长的。  
SSM的原理和RNN类似（TODO：学习研究下，我还没有太想明白其中的相似性）  
#### S6对比S4——选择机制理解  
作者提出的Selective SSM即S6，相比S4的创新点在于引入选择机制。这里比较抽象了，大致讲讲理解，作者认为Transformer是相当于不压缩上下文信息，所以效果好但开销大。对于SSM模型这样的递归模型而言，实际做的是对上下文的压缩，选择机制就是一种信息压缩的方法，对上下文压缩质量高才能有好的效果。但如果为了并行计算、保持线性时不变的简单结构，那就是恒定的方式从上下文中选择信息，肯定是不好的。所以一定要改掉线性时不变，要进行过滤。所以S6做的改变具体而言就是把固定的参数改为输入的函数，使得不同的输入会对应不同的网络参数，能够更有“选择性”地去过滤内容。另一种角度去描述选择机制就是系统变为受输入数据影响的data-controlled/data-gated。  
大致研究了下，感觉作者并没有严格证明 **为什么固定参数的系统无法实现好的长期记忆？**，像是一种很牛的数学直觉。很多种抽象理解的方式，也有很多人讨论，想办法去想通吧。  
但是变为线性时变的系统就会导致结构不再简单，不好并行化。但是作者这里设计了Hardware-aware state expansion算法，从而使得可以并行化，依然可以高效的推理和训练。  
用通俗一点的方式来理解，Transformer就像是一个极佳记忆力的机器，记得所有信息，但是只能记一个不大的窗口区间，一旦超了就全忘了。而Selective SSM就像是人类一样，在不断遗忘不断记忆，但是会有选择地选择其中重要的信息保留下来，时间越长对于记忆的内容概括越厉害。  
#### S6具体计算  
TODO：论文有个结构图要贴过来  
整个过程还是比较复杂的，详细的过程需要对照代码去看（TODO）。  
### Mamba Block  
TODO：论文有个结构图要贴过来  
TODO：计算过程就对照代码看吧  
借鉴了SSM的经典H3架构。H3架构应该是有linear attention block线性注意力块和MLP。Mamba块是把这两个部分融合了，融合的思路又是借鉴的gated attention unit（GAU，门控注意力单元）。TODO：要理解这一套操作还是需要去读一下H3和GAU的设计理解原理。  
大致流程是输入分两路（支路应该是准备处理处理做残差），干路支路都会先做线性映射，然后干路上先1D卷积，再送入S6计算，最后加支路结果。融合后的结果再做线性映射输出。  
### Mamba Model  
使用Mamba Block搭模型，网络结构不算是很难理解，给了做语言模型的例子，就是堆叠多层mamba block，然后加上残差和归一化层，代码非常简单。  
  
## 代码  
https://github.com/state-spaces/mamba  
主要计算代码在mamba_ssm目录下  
mamba_ssm/ops/selective_scan_interface.py 是Selective SSM结构的代码  
mamba_ssm/modules/mamba_simple.py 是mamba block的代码  
mamba_ssm/models/mixer_seq_simple.py 是mamba模型的代码  
  
看到有另外一个简化版本的mamba代码推荐学习，可读性更好  
https://github.com/alxndrTL/mamba.py  
  
预训练的模型在hugging face上可以下载了用，在pile数据集上训练的（2020年的825G的英文语料库）。作者搞了5种大小的可以按实力和需求选用，最大2.8B的看起来也不是很夸张，推理时按32bits的1B参数=4G的话，这里单卡也能推理。小一点参数的单卡也能训练。  
| Parameters | Layers | Model dim. |   
|------------|--------|------------|  
| 130M       | 24     | 768        |  
| 370M       | 48     | 1024       |  
| 790M       | 48     | 1536       |  
| 1.4B       | 48     | 2048       |  
| 2.8B       | 64     | 2560       |  
https://huggingface.co/state-spaces  
  
### 源码安装  
官方源码要求的环境是 Linux系统、NVIDIA GPU、PyTorch 1.12+、CUDA 11.6+、python 3.7+  
然后官方自己编译的环境都是CUDA11.8/12.2  
README.md说直接pip install mamba-ssm即可，我不是很成功，卡在building了，也看到有其他人反馈这样可能卡住或报错，https://github.com/state-spaces/mamba/issues/55还讨论了挺多的。  
也有好心人打好了docker分享了也可以用。  
当然还是源码安装好，自己试了下可以装上。  
  
环境是用的AutoDL的4090主机，预装镜像是Miniconda conda3+Python 3.10(ubuntu22.04)+Cuda 11.8，已有conda和cuda了，其他用conda搞的环境。  
  
    conda create --name mamba1 python=3.10  
    conda init bash && source /root/.bashrc  
    conda activate mamba1  
    conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda==11.8 -c pytorch -c nvidia  
    pip install packaging  
pytorch安装那一步还是会比较慢看网速，会装很多包，PyTorch 1G+，估计十多分钟的样子。  
* cuda安装版本问题  
其中pytorch-cuda是仅适用于pytorch的cuda库，说是conda上可能只有一个单独的库，实际不会把nvcc等cuda基础库安装了。如果有问题，需要完整安装各种平台都适用的cuda库应该是装cudatoolkit==11.8。但是这里遇到过一些包装下来还是有问题装了最新版……可能是源的问题，官网说安装命令用conda install nvidia/label/cuda-11.8.0::cuda-toolkit，但是安装了之后其中几个包包括cuda-nvcc还是最新版，有毒……继续conda install nvidia/label/cuda-11.8.0::cuda-nvcc（以及cuda-cuobjdump cuda-cuxxfilt cuda-compiler cuda-nvpurne几个包，都是手动解决的……应该可以直接conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit解决）  
有一次还遇到比较坑的事情，就是nvidia库里不知道为啥还装了个cuda-version的包获取版本信息，结果获取的是12.5……这种情况要再单独安装cuda-version=11.8  
* numpy兼容问题  
还遇到的情况是pytorch旧版不兼容numpy2，回退numpy版本，例如装numpy==1.24.3  
* 编译器版本问题  
编译casual_conv1d说gcc 11之后的版本不支持……conda install gcc=10  
  
  
这样基本环境就有了，然后在源码目录下  
  
    python setup.py install  
就可以安装，但这里有个小坑是装依赖项的时候还是拉的官方源，国内机器非常慢会断连接中断安装……一开始一个一个手动pip，后面发现可以搞个setup.cfg配置文件写上  
  
    [easy_install]  
    index_url=https://mirrors.aliyun.com/pypi/simple/  
就可以了。当然整个安装过程还是比较耗时的，大部分时间都是在编译源码csrc/selective_scan.cpp的C++写的算法，感觉花了差不多半个小时。软件源网络没问题装transformer、causal_conv1d等几个依赖包应该还是没那么久。  
然后就可以from mamba_ssm import Mamba验证是否安装成功。没问题的话就可以下载Hugging Face上预训练的基于Mamba的语言模型跑了。  
但是没太多应用性的东西可以跑起来玩，只有两个评测代码可以跑。一个是evals目录下有跑语言模型评估工具库lm-evaluation-harness的代码，一个是benchmarks目录下有跑推理速度测试的代码。  
  
## Mamba与视频  
### 研究思路  
Mamba的优势在于长上下文，所以针对高分辨率图像/视频的场景应该是有用的，一般模型没办法把整个图像/视频丢进去，能拿到更多全局信息进行分析。  
我理解用于CV任务的关键是通过扫描等方式，将2D图片转为1D序列送入Mamba网络，后面的模型训练应该相比做NLP任务没有太大需要调整的。  
但是2D转1D序列如何保留图像&视频的空间关系？这是个大问题。不像CNN本身卷积核就按空间关系运算的。即使对比ViT的2D转1D方式，由于无法像Transformer一样处理完整的一个上下文窗口，Mamba更接近RNN的迭代会不断遗忘，所以也要有一些方式来弥补空间关系的损失。  
### 论文整理  
搜集的思路是  
* 搜索Vision/Image/Video+Mamba关键字  
* 从Vision Mamba和VMamba两个基础CV Mamba模型文章查找引用  
* 从U-Mamba 医学图像分割开坑文章查找引用  
  
标题标记方式为 **“首发日期 机构-内容-方向（-重要性）”**  
由于医学任务数量较多，参考价值有限，先划去降低优先级处理；其他一些类似的任务文章页划去降低优先级处理  
共计37篇，其中：  
* 14篇医学相关忽略其中13篇（有一篇轻量化可以看看思路）  
* 4篇基本模型重点学习（Vmamba、Vision Mamba、Mamba-ND、Local Mamba）  
核心的是：最初将Mamba用于CV任务的两个奠基模型：Vmamba、Vision Mamba  
* 9个图像相关任务（9个各不相同很绝，3D分割、去雾、生成、全色图像锐化、重建、分类、目标检测、编辑、超分）  
* 5篇视频相关（1篇视频生成，1篇手势识别，1篇医学视频分割，2篇视频理解是同一个模型重点看下）  
* 3篇点云相关（都是点云分割+分类）  
* 1篇特殊序列（人体动作生成）  
* 1篇理论（性能测试）  
  
#### ~~（24.1.9 多大-医学图像分割Mamba-医学图像分割-开坑）U-Mamba Enhancing Long-range Dependency for Biomedical Image Segmentation~~  
图像分割常用的网络U-Net其实很值得学习下，简单理解其核心思想是Encoder是多层下采样获得特征，Decoder是逆过来多层上采样恢复出语义分割图（一般都是4层），而Decoder每一层的输入不仅是上一层，而且加入Encoder中同层的输入，解决了下/上采样中带来的一些损失效果很好。  
#### （24.1.17 华科-CV Mamba-基本模型）Vision Mamba Efficient Visual Representation Learning with Bidirectional  
训练是在ImageNet-1k分类数据集上预训练，然后在不同具体任务的数据集上微调（包括语义分割、目标检测、实例分割）  
对标的是DeiT，性能提升其实不太明显，但是差不多同性能跑1248\*1248的图像任务（6084个token，8卡A800服务器），可以节省86.8%的显存，速度上也快了2.8倍。  
用的是双向mamba块加position embedding的方法，代码没太细细研究大致看了下，是基于mamba1.1.1版本改了下block结构分两路做正反向的扫描，后续使用还是直接block堆叠挺清楚的。  
TODO!!!!!  
#### （24.1.18 国科大-CV Mamba-基本模型）VMamba Visual State Space Model  
TODO!!!!!  
比较Vision Mamba和VMamba，二者基本的通过扫描的方式将2D图像转1D序列并保持2D图像位置关系信息的思路应该差不多。  
但是VMamba似乎做的更复杂一些。网络架构上，VMamba是比较特殊的，没有保持原本Mamba网络的架构（统一的Mamba块堆叠）。首先是也改了扫描方式，对于一个patch都不是简单扫描一遍展开了，而是搞了CSM（交叉扫描）从左上角行、左上角列、右下角行、右下角列扫了4遍，所以用了4倍内存但是应该空间信息会更多一些。TODO：这里代码我没太看懂……对于各个patch是怎么处理的？然后结合S6加上卷积啥的合成一个VSS Block。然后模型是用了4层的VSS Block，每一层都会有下采样使得特征的尺寸不断缩小。这里的做法我开始很不理解，后面感觉是和U-Net比较相似的网络结构，应该是搞图像分割的比较熟悉。  
看起来VMamba比VMamba的引用和代码star都要略少一点点。  
#### （24.1.25 港科-视频分割Mamba-视频分割-首个视频模型）Vivim a Video Vision Mamba for Medical Video  
TODO!!!!  
TODO：确认如何视频输入（应该就是扩展到时域之后3D扫描）  
应该是和VMamba那种U-Net结构比较相似，用Mamba块做了4层降采样的方式。  
只用Mamba做了Encoder，Decoder部分就是直接轻量级CNN做的。TODO：为什么不用Mamba做Decoder？  
#### ~~（24.2.4 上交-医学图像分割Mamba-医学图像分割）VM-UNet Vision Mamba UNet for Medical Image Segmentation~~  
#### （24.2.5 港中深-3D图像分割Mamba-3D图像分割）nnMamba 3D Biomedical Image Segmentation, Classification and Landmark Detection with State Space Model  
#### ~~（24.2.5 意大利理工-对Mamba性能的一些测试-模型测试分析）Is Mamba Capable of In-Context Learning~~  
#### ~~（24.2.5 中科院深圳-医学图像分割Mamba-医学图像分割）Swin-UMamba Mamba-based UNet with ImageNet-based pretraining~~  
#### （24.2.6 南京科技大学-图像去雾Mamba-图像去雾）U-shaped Vision Mamba for Single Image Dehazing  
#### ~~（24.2.7 牛津-医学图像分割Mamba-医学图像分割）Mamba-UNet UNet-Like Pure Visual Mamba for Medical Image Segmentation~~  
#### （24.2.8 UCLA-多维输入Mamba-基本模型）Mamba-ND Selective State Space Modeling for Multi-Dimensional Data  
TODO!!!!!  
确认是否有新的图像/视频输入Mamba方式  
#### （24.2.8 美团-图像生成Mamba-图像生成）Scalable Diffusion Models with State Space Backbone  
#### ~~（24.2.11 牛津-医学图像分割Mamba-医学图像分割）Semi-Mamba-UNet Pixel-Level Contrastive Cross-Supervised Visual Mamba-based UNet for Semi-Supervised Medical Image Segmentation~~  
#### ~~（24.2.13 广州上交AI医疗研究中心-医学图像分割Mamba-医学图像分割）P-Mamba Marrying Perona Malik Diffusion with Mamba for Efficient Pediatric Echocardiographic Left Ventricular Segmentation~~  
#### （24.2.16 华科-点云分析Mamba-点云处理）PointMamba A Simple State Space Model for Point Cloud Analysis  
#### ~~（24.2.16 牛津-医学图像分割Mamba-医学图像分割）Weak-Mamba-UNet Visual Mamba Makes CNN and ViT Work Better for Scribble-based Medical Image Segmentation~~  
#### ~~（24.2.19 中科院-合肥全色图像锐化Mamba-全色图像锐化）Pan-Mamba Effective pan-sharpening with State Space Model~~  
#### （24.2.23 清华深研院-图像重建Mamba-图像重建）MambaIR A Simple Baseline for Image Restoration with State-Space Model  
#### （24.2.24 台大-食品图像分类Mamba-图像分类）Res-VMamba Fine-Grained Food Category Visual Classification Using Selective State Space Models with Deep Residual Learning  
#### （24.3.1 武大-点云分析Mamba-点云处理）Point Could Mamba Point Cloud Learning via SSM  
#### （24.3.4 阿里-红外小目标检测Mamba-目标检测）MiM-ISTD Mamba-in-Mamba for Efficient Infrared Small Target Detection  
#### ~~（24.3.6 广州医科大-医学图像分类Mamba-图像分类）MedMamba Vision Mamba for Medical Image Classification~~  
#### （24.3.8 NEU-图像编辑用了点Mamba-图像编辑）InstructGIE Towards Generalizable Image Editing  
#### （24.3.8 北大-轻量级医学图像分割Mamba-医学图像分割-轻量级方向好思路）LightM-UNet Mamba Assists in Lightweight UNet for Medical Image Segmentation  
TODO!!!!  
分析轻量化的方向  
#### ~~（24.3.8 港中文-内窥镜运动引导Mamba？-医学很专的方向）Motion-Guided Dual-Camera Tracker for Low-Cost Skill Evaluation of Gastric Endoscopy~~  
#### ~~（24.3.11 港科大-计算病理学Mamba-医学很专的方向）MambaMIL Enhancing Long Sequence Modeling with Sequence Reordering in Computational Pathology~~  
#### （24.3.11 上海AI Lab-视频理解Mamba模型-视频理解-第二个视频模型）VideoMamba State Space Model for Efficient Video Understanding  
TODO!!!!!!  
TODO：确认做的质量如何，是否可以作为代码基础去改进？（目前认为是可以的）  
分析了Vision Mamba和VMamba，认为Vmamba做了降采样是不标准的，坚持按照ViT的模型就是直接堆叠不做降采样。说和Vision Mamba的实现补交相似，有一些细节简化，包括中间的[cls] token认为不需要，rotary position embedding不需要。（TODO：为什么）  
实现原理上也确实比较直接没太多可说的，还是加上时域3D双向扫描，不过讨论了下扫描的方式，看是先空域再时域、先时域再空域、正向先空域再时域反向反过来、扫四遍正向反向先时域先空域都跑，最后结论是简单的先空域再时域就挺好的，这个结论我很喜欢。  
还有个性能上的创新点说是参考了UMT做了Mask Modeling。（TODO：这个我没有看懂……）  
文章写法上没有特别去强调模型创新性，毕竟Vision Mamba已经有了，扩展到Video也并没有什么本质上的创新。但文章很出彩的地方是实验做的不错很全面，强调了VideoMamba开创性工作的意义是实验证明了Mamba在视频任务中的四个优势，分了四个部分实验来说明  
* 1.证明VideoMamba可以扩大规模  
首先跑了ImageNet-1K效果不错，并且使用了自蒸馏的方式用小模型去辅助微调更大的模型，以此证明VideoMamba性能不错并且可以扩大规模。（TODO：小模型辅助大模型训练的方法可以学习下，只听说过大模型蒸馏小模型）  
* 2.证明VideoMamba对短视频敏感性好  
跑了短视频理解数据集Kinetics-400和Something-Something V2。  
* 3.证明VideoMamba对长视频理解潜力大  
跑了长视频理解数据集Breakfast、COIN、LVU。可以小参数量跑到SOTA。  
* 4.证明VideoMamba还兼容多模态  
跑了视频-文本的5个多模态数据集也不错。  
#### （24.3.11 上交-点云分析Mamba-点云处理）Point Mamba A Novel Point Cloud Backbone Based on State Space Model with Octree-Based Ordering Strategy  
#### （24.3.12 东京大学-视频生成Mamba-视频生成）SSM Meets Video Diffusion Models Efficient Video Generation with Structured State Spaces  
TODO!!!  
是否和视频处理任务有关？  
#### ~~（24.3.12 莫纳什大学-长序列动作生成Mamba-动作生成）Motion Mamba Efficient and Long Sequence Motion Generation with Hierarchical and Bidirectional Selective SSM~~  
#### ~~（24.3.12 浙大-医学图像分割Mamba-医学图像分割）Large Window-based Mamba UNet for Medical Image Segmentation Beyond Convolution and Self-attention~~  
#### （24.3.13 西安交通大学-图像超分Mamba-图像超分-改进Vision Mamba）Activating Wider Areas in Image  
TODO!!!  
确认改进点  
#### ~~（24.3.13 中科院成都-医学放射剂量预测Mamba-很专的医学方向）MD-Dose A Diffusion Model based on the Mamba for Radiotherapy Dose Prediction~~  
#### ~~（24.3.14 南京大学-医学图像分割Mamba-医学图像分割-升级VM-UNET）VM-UNET-V2 Rethinking Vision Mamba UNet for Medical Image Segmentation~~  
#### （24.3.14 清华深研院-手势识别Mamba-手势识别）MambaTalk Efficient Holistic Gesture Synthesis with Selective State Space Models  
TODO!!!  
研究下输入是否和视频相关  
#### （24.3.14 上海AI Lab-视频理解Mamba套件-视频理解-也是第二个视频模型更多代码）Video Mamba Suite SSM as a Versatile Alternative for Video Understanding  
TODO!!!!!!  
确认套件代码能做什么  
#### （24.3.14 悉尼大学-改Vision Mamba扫描方式-基础模型-改扫描方式这个好思路）LocalMamba Visual State Space Model with Windowed Selective Scan  
TODO!!!!!  
了解扫描方式改进的方法  
  
### 代码  
#### Video Mamba  
源码部署测试中，还未能确认成功……  
一些问题记录下：  
* 基本环境  
Torch 2.1.2 CUDA11.8 要手动先装好causal-conv1d和mamba，看有博客上是自己找的causal-conv1d和mamba的版本安装的，就会存在没有修改成双向mamba “bimamba”的问题，需要自己找到python3.10/site-packages/mamba-ssm/目录装好的代码去手动用VideoMamba里的修改后双向mamba去替换，反正有点麻烦~直接git clone VideoMamba源码，然后安装。  
```  
（这里省略了Mamba基本环境的安装，参考上文的内容，略过）  
git clone https://github.com/OpenGVLab/VideoMamba.git  
cd VideoMamba/causal-conv1d/  
python setup.py install  
cd ../mamba  
（需要改国内源别忘了创建setup.cfg）  
python setup.py install  
```  
* 其他依赖包  
```  
cd VideoMamba  
pip install -r requirements.txt  
```  
但是其中有两个包是有问题的，需要手动安装，修改下requirements.txt文件注释掉apex==0.1和skimage=0.0  
用到的包不少，还有TensorFlow包600M，一起也有1G+，看网速了。  
* apex==0.1包安装  
这个包是PyTorch用NIVDIA显卡做混合精度&多卡并行的扩展库  
这个包直接pip安装似乎是不行，没有对应版本……可以到github下载源码然后安装似乎就是0.1版本  
  
```  
git clone https://github.com/NVIDIA/apex  
cd apex  
# if pip >= 23.1  
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./  
# pip < 23.1  
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./  
```  
这里因为也是要编译C++，比较花时间。  
* skimage=0.0包安装  
应该是requirements.txt文件里写错了，这个包是scikit-image  
  
##### 单卡跑训练  
这里以video_sm单模态中kinetic-400任务为例，其他应该差不多  
  
* 启动脚本移除分布式训练部分  
如果按源码中准备的shell脚本去跑，需要是有slurm服务器，并且单机8卡以及2台8卡的环境……  
只有单卡测试验证的话需要调整脚本，以最小的 run_f8x224.sh脚本为例，删除其中srun启动部分，然后调整其中数据集路径（看了代码data_path是标签文件地址，prefix是视频文件地址），设置日志地址和输出模型地址log_dir和output_dir，batch_size和epoch根据情况设置，num_workers也可能调整。另外验证的话可能跑不出best checkpoint，test_best也可以删掉。  
```  
python run_class_finetuning.py \  
        --model videomamba_tiny \  
        --data_path '/root/autodl-fs/renyu_kinetics-2' \  
        --prefix '/root/autodl-fs/renyu_kinetics-2' \  
        --data_set 'Kinetics_sparse' \  
        --split ',' \  
        --nb_classes 400 \  
        --log_dir "/root/autodl-tmp/videomamba_run/log" \  
        --output_dir "/root/autodl-tmp/videomamba_run/output" \  
        --batch_size 1 \  
        --num_sample 2 \  
        --input_size 224 \  
        --short_side_size 224 \  
        --save_ckpt_freq 100 \  
        --num_frames 8 \  
        --num_workers 1 \  
        --warmup_epochs 5 \  
        --tubelet_size 1 \  
        --epochs 7 \  
        --lr 2e-4 \  
        --drop_path 0.1 \  
        --aa rand-m5-n2-mstd0.25-inc1 \  
        --opt adamw \  
        --opt_betas 0.9 0.999 \  
        --weight_decay 0.1 \  
        --test_num_segment 4 \  
        --test_num_crop 3 \  
        --dist_eval \  
        --test_best \  
        --bf16  
  
```  
  
*  准备数据集  
还要准备数据集sthsthv2或者kinetics-400，都不好整。可以随便看下作者团队提供的Kinetics-400的格式，就是一个目录下有26w+的视频，然后放上train.csv val.csv test.csv三个文件，文件格式很简单，第一列文件名，第二列是分类序号，所以可以自己按照格式做小的示例数据放上去。注意--data_path '/root/autodl-fs/renyu_kinetics-2' 参数是.csv文件目录，--prefix '/root/autodl-fs/renyu_kinetics-2' 参数是视频文件目录。  
* 准备预训练模型  
还要下载ImageNet的预训练模型，因为video-sm的模型都是基于ImageNet预训练模型去微调的。参考https://github.com/OpenGVLab/VideoMamba/blob/main/videomamba/image_sm/MODEL_ZOO.md。按说要下载tiny、small、middle三个，不过只是要能跑的话理论上搞一个也行。然后记得修改videomamba/video_sm/models/videomamba.py里面的MODEL_PATH变量为你放预训练模型的目录。  
* 处理预加载模型state_dict层级问题  
load_state_dict出错，做个state_dict = state_dict['model']似乎可以跑了……]  
* 移除train_one_epoch中检查分布式损失函数计算的代码  
这一段代码必须是分布式执行的，单卡都没初始化相关变量，但是单卡也不需要检查损失函数是否NaN或者Infinite，直接全部注释掉  
```  
        loss_list = [torch.zeros_like(loss) for _ in range(dist.get_world_size())]  
        dist.all_gather(loss_list, loss)  
        loss_list = torch.tensor(loss_list)  
        loss_list_isnan = torch.isnan(loss_list).any()  
        loss_list_isinf = torch.isinf(loss_list).any()  
  
        if loss_list_isnan or loss_list_isinf:  
            print(" ========== loss_isnan = {},  loss_isinf = {} ========== ".format(loss_list_isnan, loss_list_isinf))  
            print("Loss is {}, stopping training".format(loss_value))  
            sys.exit(1)  
```  
* 移除评估过程中分布式同步方法  
在main函数训练主循环跑完测试最佳模型（开了test_best参数）或者直接eval模式跑的。会调用分布式进程阻塞方法等待所有进程结束同步  
```  
    torch.distributed.barrier()  
```  
这一句也直接注释掉。  
  
由于提供的原代码是在多显卡的服务器上通过srun提交训练任务，并且搭配了ImageNet-1k、Kinetics-400这些比较大的数据集，没能直接跑起来，后续还是需要读代码看下如何单GPU运行并且跑样本做推理测试，以及如何做的预训练和微调。
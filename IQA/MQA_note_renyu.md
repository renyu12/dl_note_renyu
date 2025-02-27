# 背景  
考虑多模态输入的质量评估任务，主要考虑音视频输入，也可以考虑文本输入  
输出主要考虑音频+视频的视听质量评分，也可以考虑有单模态质量评分的输出  
  
# 小于3模态质量评估  
多模态的研究还不少，但是三模态VQA确实少  
主要是理解和生成  
多模态检索  
多模态字幕  
多模态问答  
## 视频质量评估  
经典VQA不说了  
## 音频质量评估  
分为音频（音乐类）和语音（通话）两个类别，评估标准是不一样的。  
## 文本质量评估？  
单独好像没啥质量评估任务，有点奇怪  
参考下Legit: Text Legibility For User-Generated Media，分析文本可读性  
## 视频+音频相关任务  
主要是理解和生成  
事件检测、定位  
音频辅助视频理解、动作识别  
音频检索视频，视频检索音频  
音频/声源分离（不同说话人、乐器）  
视频生成音频或音频生成视频  
AVQA  
## 视频+文本相关任务  
caption内容描述（事件、动作、场景）  
视频问答VQA Question Answering  
视频检索Retrieval 文本检索视频，视频检索文本  
生成字幕、翻译（感觉还是音频部分）  
视频摘要Summarization（概括核心信息）  
分类与标注勉强也算吧？  
视频中文本检测与识别  
文生图  
内容审核  
## 音频+文本相关任务  
语音识别ASR  
情感分析（加上语气了）  
对话分析  
音频与文本检索 音频检索文本，文本检索音频  
语音生产TTS  
音频内容描述，内容摘要  
  
# 思路  
## 多模态重要性建模  
**不同模态在不同场景下重要性不同**  
区分场景：  
电影、电视剧、综艺、动画片、新闻、秀场、教学都重要  
游戏、体育、风光画面重要  
语音、音乐、语言类节目、知识、无字幕播客  
有字幕外文播客  
不同模态权重要动态分配  
### 分析方法  
消融实验  
合成失真  
## 音频模态处理  
视频帧与音频特征对齐  
是否要评估不同失真类型和失真级别  
音画不同步失真？  
UGC视频有语音、音乐、环境音，还有一些复杂的情况如特效音、噪声、动作声、动物声、机械声等，很难覆盖  
## 文本模态处理  
什么样的文本？  
* Caption  
说明字幕、语音字幕、翻译字幕  
* Transcript  
语音的逐字记录  
* 视频中文本信息OCR  
* 元数据  
标签、标题、作者说明  
* 视频摘要/描述  
* 用户评论  
字幕不匹配问题？时间怎么对？  
## 多模态融合  
**网络结构、特征对齐**  
## 数据集搭建  
**大数据可以解决一切问题，但没有怎么办**  
保证版权  
具备三模态信息  
场景丰富  
不同质量（是否需要合成？）  
质量评分（三模态+整体MOS）  
统一实验环境/在线众包？  
参与者专业？非专业  
标注（失真类型、级别，场景分类）  
模态权重？  
数据量 500？1000？  
### VQA数据集检查  
* CVD2014  
234个视频，太少了不要  
* LIVE-Qualcomm  
208个视频，太少了不要  
* KoVNiD-1k  
2018年，1200个视频，一个8s  
有声音可以搞  
* LIVE-VQC  
585个视频  
有声音可以搞  
* Youtube-UGC  
1000+视频  
没声音  
* LSVQ  
39000个视频  
没声音  
* MSU-VQMB  
就36个视频，不过都是比较高清的。  
总共收集了来自 10,800 个参与者的 766,362 个有效评价。  
* DIVIDE-3k  
2022年Weisi Lin老师团队考虑审美分数新建的数据集，3634个训练视频，909个验证视频。  
  
# 数据集  
## AVQA  
近期的很少，主要是上交在做  
| 数据库 | 独特的 A/V 内容 | 总音频/视频 | 失真类型 | 视频长度 |  
| --- | --- | --- | --- | --- |  
| SJTU-UAV | 520 | 520 | 野外 | 960×720-1920×1080 | 8秒 |  
| LIVE-SJTU | 14 | 336 | 视频压缩、音频压缩 | 1920×1080 | 8秒 |  
| UnB-AVC | 6 | 72 | 视频压缩、音频压缩 | 1280×720 | 8秒 |  
不过早期也有一些，例如INRS数据集等，都是参考视频很少，做了多级音视频失真的FR AVQA数据集  
## 一些音频相关任务  
### AQA  
早期有很多是关于语音的，用于优化通话质量的  
P.800 Subjective Speech Quality Database（ITU-T）  
VoiceBank-DEMAND（用于语音增强的音频质量评估）  
AV-VoIP（音视频通信质量评估数据集）  
DNS Challenge Dataset（用于噪声抑制和语音质量评估）  
### 音视频中音频分类  
2017 AudioSet  
2020 VGGSound  
都是YouTube上的视频，打了音频分类标签  
## 一些视频相关下游任务  
### 视频理解  
MSR-VTT  
YouCook2  
ActivityNet Captions  
### 视频检索  
COCO-Retrieval  
### 视频字幕  
COCO-Caption  
### 视频问答  
VQA v2  
## 下载  
涉及YouTube视频用yt-dlp库  
比较棘手的问题是现在YouTube对于视频下载限频很严格，发现并发请求后很可能会封禁账号一段时间导致无法访问视频资源，需要控制请求频率慢慢下载  
  
# 论文整理  
## 音频+视频VQA  
### （19.10.24上交LIVE-SJTU数据集）Study of Subjective and Objective Quality Assessment of Audio-Visual Signals  
可以说是第一个AVQA数据集。CDVL数据集中找到14个参考视频，都是质量很高的，考虑2类x4个级别的视频失真，同时可以搞3个级别的音频压缩失真，所以一共是24种失真，得到336个失真视频。35个受试者（基本是Texas大学的研究生）  
比对了一些比较直接的方法，包括  
* 分开视频&音频评分相乘  
* 视频&音频特征用SVR/随机森林  
* 音频特征提取部分仿照经典VQA指标SSIM、VMAF等转1D做AQA  
* 神经网络提取视频&音频特征  
  
其实都是特征提取加回归，主要是特征提取部分换了点方法，还都是比较传统的一些特征，除了最后神经网络。回归头基本都是SVR  
  
  
### （21.9.19pub上交）Deep neural networks for full-reference and no-reference audio-visual quality assessment  
比较早的工作，做的还比较粗糙，其实延续了19年的工作用DNN提取视频和音频特征的部分，不过这里算是第一次用端到端的DL方法了，还特别接了个GRU。更早的工作还是特征提取后做SVR、随机森林等。  
视频和音频还是分开处理的：  
* 视频路径  
使用CNN提取单个帧特征，固定了224x224的patch大小所以还做了多次crop取平均，然后过GRU考虑时域就成了空域时域都考虑的VQA（感觉是比较老的VQA做法才用RNN）  
* 音频路径  
也是把音频切小片段，测了两种特征提取方法，一是计算频谱图然后用CNN提取特征，二是用个SoundNet 1D卷积网络，反正测出来前一种一般好一点，最后也是过GRU考虑时域。  
两路获得的特征过MLP给出预测评分。  
唯一能跑的AVQA数据集就是SJTU-AVQA，之前也没啥厉害的baseline，尤其是NR任务，所以没有很多实验要对比轻松sota，实验部分就自己调调参测试下，做的比较浅。  
代码未开源。  
  
### （22.1.31上交）Attention-guided neural networks for full-reference and no-reference audio-visual quality assessment  
这篇文章应该说是19年前一篇DNN for FR&NR AVQA的小改进，基本是一样的，实验做的会更完善一些，忽略前一篇看这篇就行。  
网络结构基本没变，就是视频特征提取加了ROI，本来是随机crop，现在加了一个显著性检测的模型FES可以获得每个帧的显著性map，就会crop其中更重要的一些patch，性能提升了。  
其他多测了一个UnB-AVC数据集，测了计算效率，这样一些小完善点。  
代码说开源但是没看到。  
  
### （22.10.4上交SJTU-UAV数据集）Subjective and Objective Audio-Visual Quality Assessment for User Generated Content  
做了SJTU-UAV AVQA数据集，是从YFCC100m 数据集中选的520个视频，21名受试者评分。  
为了说明这个数据集做的丰富性比较好，对比了另外两个LIVE-SJTU和UnB-AVC数据集，还有VQA的数据集LIVE-VQC，看了5个视频属性+4个音频熟悉，发现SJTU-UAV数据集的分布就比较好一些。  
也做了个baseline的AVQA模型，基本和之前做的模型一脉相承，改进的点主要在于音频特征提取部分，不再是直接转频谱图输入CNN了，而是提取了色度图、CQT、MFCC 和 GFCC四通道之后再输入一个复杂的CNN（频域、时域、fusion三种不同参数的卷积）。还有就是把之前用的GRU都换成Bi-LSTM了。  
数据集说开源但是没看到……  
  
### （23.9.11pub上交SJTU-UAV数据集）Audio-Visual Quality Assessment for User Generated Content Database and Method  
这篇感觉和22年的SJTU-UAV数据集文章没有啥明显区别，要简短一点。baseline模型用的SVR整合多个特征的更传统的方式。看一篇就行。  
  
### （24.7.29上交）UNQA Unified no-reference quality assessment for audio, image, video, and audio-visual content  
图像IQA、视频VQA、音频AQA、音视频AVQA四种不同的QA数据集按说要训练四类不同的模型来评分，这篇文章中尝试做了一个统一的模型，根据输入的模态给出不同的评分  
还有一个问题是，即使是同一种模态的数据集之间也是没法互通的，因为绝对评分标准也不一样。但是我们认为其相对排序一样就可以做  
  
### （24.12.25pub哈工大）Enhancing No-Reference Audio-Visual Quality Assessment via Joint Cross-Attention Fusion  
就是CNN提取视频、音频特征后，用的注意力机制的网络做的特征融合，再过MLP，感觉思路挺直接的。  
只是说数据集也只有LIVE-SJTU、UnB-AVC这种几百数据量级的，上注意力的意义也就那样……数据集限制做不了太多了。  
  
# 相关论文整理  
## 音频AQA  
### （24.2.1CMU 大模型AQA）PAM Prompting Audio-Language Models for Audio Quality Assessment  
很有意思的工作，直接使用预训练的音频大模型ALM通过提示词做音频质量评估，不进行微调。  
不太熟悉AQA的指标，不过看在很多下游任务上确实有优化，是个低成本又有应用价值的事情。  
  
### （25.2.7Meta 语音音乐声音全统一美学）Meta Audiobox Aesthetics Unified Automatic Quality Assessment for Speech, Music, and Sound  
很新也很相关的工作，看起来做的相当好，可以作为AQA这边的一个里程碑了。之前的AQA专注于语音或者音乐，并且主要是做技术质量，这个工作比较可解释性的定义了音频的美学分数，分了四个维度（制作质量PQ、制作复杂性PC、内容享受CE、内容有用性CU），避免了直接单维度音频美学分数定义的混乱。  
数据标注先由专家评了黄金集作为参考，然后做众包筛去低质量评分。数据集有500小时，97000个音频，时长在10-30s之间，语音、声音、音乐基本各占三分之一，但是没有说数据来源，只说了是开源的数据集。  
模型结构没有做的很复杂，原始输入都是16kHz的音频波形（需要做预处理统一成16kHz单通但并随机采样10s，归一化），先embedding再过transformer encoder，最后分4个MLP回归头预测4个维度的回归分数。音频embedding是用的2022年的工作WavLM，Transformer就是原始的Transformer encoder，用了12层768维。似乎用了多层的输出？（没有讲清只给了公式，可能要看代码）  
这个工作做的比较新，评估是不太容易的，没啥对标的工作：  
* 单独看语音AQA还是能做  
选了三个专门的AQA模型DNSMOS、SQUIM、UTMOSv2以及PAM，只有PAM是无参通用AQA模型，主要就是对标PAM。语音数据集用的VMC22-main和VMC22-OOD，比较下来Audiobox相当强  
* 单独看音乐和音频有点难  
只有PAM对标，数据集也是PAM有个小的音乐、声音数据集对标下，这里结果有点特殊，按原始总分audiobox打不过PAM，但是重新标注了4个维度的美学分数基本是占优的，所以认为直接做音频的整体评分不合适。另外音乐上PAM占优，这里认为是PAM-music数据集用到了一些生成音乐，而audiobox都是真实音乐  
* 语音+音乐+声音的美学评估  
由于目前没什么公开的其他类似数据集可以做评估，他们自己有从其他数据集中找原始音频标注了一个测试集，对标前面几个模型证明了可以吊打。  
  
还考虑了应用价值，测试了用模型的美学评分按百分比过滤下游任务（文本转语音/音乐/声音）的训练数据集，可以提升性能。  
  
## 音频Encoder模型  
### （16.9.29谷歌 VGGish）CNN Architectures for Large-Scale Audio Classification  
经典模型，在私有的YouTube-100M上训练的  
  
### （16.10.27MIT）SoundNet Learning Sound Representations from Unlabeled Video  
早期经典工作，通过在200万无标签视频上自监督训练，获得了一个音频CNN，发现效果还不错  
自监督训练的方法有点抽象，是用ImageNet上训练的CNN提取视频场景分类的分布信息，然后同一个视频的音频过音频CNN，优化目标是场景分类的分布KL散度接近视频场景分类的结果。那应该就是假设音频和视频都是在同一场景下关联的，利用这个信息来训练。听起来从视觉知识训练音频只是有点不靠谱，但是还真的可以。  
  
### （17.5.23谷歌 L3-Net）Look, Listen and Learn  
也是早期的经典工作，其实和SoundNet的思路很像，就是探索拿音视频输入去自监督预训练，学习音视频匹配的关系有啥用。和SoundNet的区别是这里用了很简单的一种AV匹配的自监督学习任务，就取得了更好的效果  
模型很简单，视频过VGG，音频也是过个CNN卷积，最后过MLP做个二分类，输出视频和音频是否匹配。结果很神奇的发现这样训练出来的音频CNN提取的特征用于音频分类任务效果非常好。  
  
### （19.12.21英国Surrey大学）PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition  
经典音频encoder，ResNet+EfficientNet的Backbone  
  
### （21.4.5MIT）AST: Audio Spectrogram Transformer  
经典音频encoder，VALOR应该用的这个，基于Transformer  
  
## 视频音频文本三模态  
### （21.4.22谷歌）VATT: Transformers for Multimodal Self-Supervised Learning from Raw Video, Audio and Text  
非常经典的VAT多模态模型，接收原始格式的视频、音频、文本输入，进行了大规模的自监督预训练  
思路也很简单，就是三个模态各自的数据经过各自的tokenization、linear projection之后输入到同一个Transformer即可，然后自监督训练的内容是对比学习对齐不同模态的，最后的性能是能够和单模态特定的更小size的Transformer差不多的。  
具体一些：  
* 视频  
Tokenization是是TxHxWx3通道的视频，划分为dTxdHxdWx3的小patch，然后做线性投影化成D维向量  
* 音频  
本来就是一维的波形，直接取dT长度做线性投影，获得D维向量  
* 文本  
先分词，然后构建一个总共v个单词的词汇表用one-hot编码，获得v维向量，再做一个有学习权重的线性投影（这是word2vec啥的？）  
  
有个小创新点是做了DropToken随机丢弃一些Token降低计算复杂度，因为视频和音频模态冗余会比较多，效果还不错  
关键的问题是如何做对比学习对齐视频和音频，做法是用NCE噪声对比估计。另外还用了多实例学习噪声对比估计MIL-NCE去对齐视频和文本，从而全部对齐。大概的原理就是同一时间时刻的token是正例要拉近距离，不同时刻的是负例要拉远距离，损失函数的设计细节要看论文  
  
### （21.6.24德国Kaiserslautern大学）AudioCLIP: Extending CLIP to Image, Text and Audio  
CLIP拓展音频输入的经典工作，做法也很直接，音频过ESResNeXt得到向量，然后CLIP是要把三模态两两之间组合都考虑到。  
TODO：训练方法上还挺多内容的  
  
### （22.3.26人大 MUSIC-AVQA数据集）Learning to Answer Questions in Dynamic Audio-Visual Scenarios  
经典的AVQA数据集，不过是音乐场景，9288个视频，22种乐器，45876个问答（在33个模板之中）。  
做了个baseline模型，音频编码器是预训练的VGGish，视频编码器就是预训练的ResNet18，文本编码器是从头训练的一个LSTM。  
但是处理流程我没看懂……TODO。这里看起来大致是视频+音频得到了空间grounding特征和时间grounding特征，然后结合问题文本在空间grounding中找关键的位置，在时间grounding中找关键时间戳，最后选出来的一部分空间&时间关键特征再连接起来过分类器去回答问题。看起来应该是比较早一点的多模态处理方法。  
  
### （23.3.12人大 CLIP4VLA）Accommodating Audio Modality in CLIP for Multimodal Processing  
一个新一些的CLIP拓展音频输入的工作，似乎考虑了音频分类标签？  
  
### （23.4.17中科大）VALOR: Vision-Audio-Language Omni-Perception Pretraining Model and Dataset  
主要是做的数据集很棒，从YouTube AudioSet 200w视频中选了100w可以用的，全部标记了视频摘要标签，并且是包含对于音频内容描述的。对比之前一些大的VAT数据集，例如HowTo100M、HD_VILA_100M、WebVid-2.5M，这些数据集确实有视频、音频、文本三模态，但是相关性没有那么好，例如文本只是转录的语音文本而非画面信息，文本光描述了画面但是没有提到音频，这样就无法实现很好的三模态对齐。  
文章中整理了经典的视频多模态预训练数据集还有下游任务的一些测试数据集可以参考。  
做了个作为baseline的三模态大模型，由视频、音频、文本三个编码器以及一个多模态decoder（用于文本生成）组成。视频编码器说CLIP和Video Swin Transformer都测了；文本编码器是BERT；音频编码器是AudioSet上预训练的Audio Spectrogram Transformer。三个模态encoder各自提取的特征要过三个各自的线性投影层去放到同一空间对齐，做特征对齐的时候考虑以文本为核心，对齐了T-A T-V T-AV，对齐的时候需要考虑全局特征投影而不是局部特征（视频帧、音频片段），所以还有全局池化啥的。  
因为还可能做文本生成的下游任务，所以预训练不仅仅只是搞对齐，不然只能拿到三个encoder做一些检索任务。还做了Captioning的预训练任务来训练一个多模态decoder，mask掉一部分视频caption，从而使得decoder可以文本生成。
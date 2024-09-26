# 整体概述  
## 是什么——图像/视频质量评估是在做什么事  
其实概念比较好理解没啥复杂的，字面意思就是评价图像/视频的质量，给出一个量化的分数  
视频就是VQA Video Quality Assessment，图像就是IQA Image Quality Assessment，现在还会做点云质量评估PCQA Point Cloud Quality Assessment  
但是难在这个“质量”是比较抽象的，一般不太指审美的评分（这个更难），而是所谓的“画质”好不好。这个评估生活中更多的是主观感受难以量化，所以理解这个任务是做什么，先要理解主观&客观VQA/IQA的定义  
### 背景补充——主观&客观质量评估  
人主观评价就是主观质量评估，不同人的评估标准都不统一，是非常难量化的。目前研究更多依赖大量人主观评价的数据库的统计数据作为标准，算是一个统计上合理的量化标准  
计算机通过数学模型给出量化评价就是客观质量评估  
做QA方向一个根本的目标就是自动生成和主观质量评估一致的客观质量评估，说人话就是让机器打分和人打分差不多  
## 为什么——QA的意义  
为什么要让机器和人打分差不多？因为有一些活就是需要机器来做的，乙方（机器）做事情不可能一遍一遍去问甲方（人）的标准，而标准不一就会导致乙方做的甲方不满意又打回重做，所以尽量一开始就统一标准。具体来说有以下一些场景需要VQA/IQA  
* 指导图像和视频编码  
有损压缩要尽量少损失画质，那怎么评价画质的损失？就需要用QA的量化指标  
* 指导图像处理算法  
怎么评价一些图像去雾、去噪算法做的好不好，也需要有量化的指标  
* 评估视频相关应用服务质量  
这里主要是VQA了，网络传输、设备故障造成的一些视频卡顿、花屏问题也需要通过VQA检测出来  
  
## IQA问题分类  
* 全参考 Full Reference-IQA, FR-IQA  
直接有原始（无失真）图像可以作为参考对比，很好做而且研究成熟  
  
* 半参考 Reduced Reference-IQA, RR-IQA  
只有原始图像的部分信息/提取的特征，介于FR和NR之间。定义有一点模糊，应该对图像做了变换或者特征提取啥的也能叫做RR，相关研究少一些。  
* **无参考/盲参考 No Reference-IQA/Blind IQA, NR-IQR/BIQA**  
最难做，但是也最有研究意义。  
还可以分为特定失真类型质量评估（例如针对模糊/块效应/噪声等失真的）和通用的质量评估。  
  
## VQA和IQA区别  
VQA以IQA为基础，但除此之外还引入更多需要考虑的。  
TODO  
  
# 具体做法  
## 关于评估指标  
### 主观评估常用指标  
MOS Mean Opinion Score 平均主观得分，就是一个量化的得分，根据大量人主观评分测试统计而来，不同数据集的得分取值范围可能不统一，还需要做映射等处理  
DMOS Differential Mean Opinion Score 平均主观得分差异，就是MOS的标准差/方程，看看主观评分争议大不大  
### 主观客观差异常用指标  
评价客观的QA算法好坏，就是要和主观评估的分数对照，这里也需要有量化的指标  
* LCC/PLCC  
Pearson Linear Correlation Coefficient，皮尔森线性相关系数，预测IQA算法准确性，结果越接近1边分数越接近  
$PLCC=\frac{\sum_{i=1}^N(y_i-\bar{y})(\hat{y_i}-\bar{\hat{y}})}{\sqrt{\sum_{i=1}^N(y_i-\bar{y}^2)}\sqrt{\sum_{i=1}^N(y_i-\bar{\hat{y}}^2)}}$  
* SRCC/SROCC和KRCC/KROCC  
Spearman rank-order correlation coefficient 斯皮尔曼秩相关系数  
$SRCC=1-\frac{6\sum_{i=1}^N(v_i-p_i)^2}{N(N^2-1)}$  
Kendall rank-order correlation coefficient 肯德尔秩相关系数  
预测IQA算法单调性，即预测评分结果的排名是不是和主观评估结果的排名一致，按顺序从1到结尾的，完全符合为1，完全倒序为-1，乱序的根据程度在[-1,1]之间  
### 全参考常用指标  
* PSNR  
Peak-to-Peak singal-to-noise ratio 峰值信噪比  
最经典的指标，计算简单非常高效  
基本的思路就是计算原始图像和失真图像之间灰度差异的均方误差MSE，但是这个MSE就是一个绝对的标量难以评判是大还是小，那就用最大像素值平方（一般255，即信息）和MSE（即噪声）做比然后取log得到信噪比。越大越好，一般认为低于30dB就失真比较严重  
$PSNR=10\log_{10}\frac{255^2}{MSE}$  
* SSIM  
Structural Similarity Index 结构相似性  
滑动窗口去对比原始图像和失真图像的亮度&对比度&结构差异，因为不只是考虑了像素点差异，对于压缩失真等像素差距小但结构上差距大的，一般能比PSNR取得更好的效果  
其他还衍生出一些优化版本，例如MS-SSIM（多尺度缩放后做SSIM），更大的计算开销可以获得更好的效果。  
* VMAF  
是Netflix提出的一个开源的全参考视频质量评估工具，给出一个0-100的质量评分。  
原理是先提取了三个视频质量指标：视觉信息保真度（VIF。这里我查到并不是直接比的原始图像和失真图像，而是一起经过一个HVS模型模拟人眼视觉感受之后再比较失真，会更符合人眼的感觉）、加性失真测量（ADM。这里我查到有小波变换啥的去分析细节的加性失真）、运动特征（Motion。大概是不只参考静帧，还要结合运动特征分析帧间预测的参考帧）。三种指标可以并行分开算提升速度，不过其中VIF一般最耗时。  
提取之后使用SVM支持向量机去融合指标，输出一个VMAF分数。这样就考虑了比较全面的质量信息，虽然是一个客观指标，但更接近人眼的感受，实际应用中效果还不错。不过由于步骤复杂，VMAF更像是一个“黑盒”指标，不好简单地描述其特性。  
* (AVQT)  
这里只是一起记录下，这个是apple 2021 WWDC上发布的一个全参考视频质量评估工具Advanced Video Quality Tool。效果还不错，有的论文里会和VMAF一起作为对比的指标。但是这是闭源的，原理未知。  
  
## 关于数据集  
### IQA数据集  
比较经典的4个常用的有参考IQA数据集列在表里  
| 数据集 | 参考图像数 | 失真图像数 | 失真类型数 | 测试人员数 |  
| --- | --- | --- | --- | --- |  
| TID 2013 | 25 | 3000 | 24 | 971 |  
| TID 2008 | 25 | 1700 | 17 | 838 |  
| CSIQ | 30 | 866 | 6 | 35 |  
| LIVE | 29 | 779 | 5 | 161 |  
无参数据集有  
* LIVE-Challenge  
2016年的一个in the wild数据集，1162张照片，8100人标注，35万份评分，MOS分数[3.42, 92.43]  
* KonIQ-10K  
2020年的一个in the wild数据集，10073张照片，1459人标注，120万份评分，MOS分数[3.91, 88.39]  
  
还有一些时间久的IVC、Toyama、A57、WIQ数据量比较小了  
TODO: 也有一些新的可能还没有完全普及作为通用标准，需要研究下，例如PieAPP, PIPAL, FLIVE, SPAQ, KADID-10k  
#### TID 2013  
这个是目前最常用最权威的数据集。都是512\*384的BMP文件。3000张=25张图片\*24种失真类型\*5个失真等级  
TODO: 具体格式  
### VQA数据集  
* CVD2014  
2014年赫尔辛基大学做的，算偏早期的数据集了。  
78 个不同的相机（手机、小型相机、摄像机、单反相机）拍摄的来自五个不同场景的 234 个视频组成  
主观评分这里好像分了组做实验，这里没看明白，可能不太容易直接拿来用。有6G、16G、23G三组数据。  
* LIVE-Qualcomm  
2017年UT Austin做的，8 种不同移动设备拍的208个视频，54个场景，一个视频15s，都是1080p的，模拟了六种常见的拍摄失真类别。  
每个视频都由 39 个不同的受试者进行评估  
* KoVNiD-1k  
2018年，1200个视频，一个8s  
质量评分是1-5五个等级。  
大概2.8G，非常小，一个视频就1M左右。  
* LIVE-VQC  
2019年UT Autstin做的，585个视频，80 个不同的用户使用 101 个不同的设备（43 个设备型号）拍摄，分辨率帧率不统一，算是比较真实丰富的。  
众包搞了205000 个意见评分，平均每个视频有 240 个人工评价，评分是0-100。  
大概5.5G，倒是比较小。  
* Youtube-UGC  
2020年Youtube提供的一个精选UGC数据集，大约1500个视频每个20s，UGC的视频包含各种类型，很强的一点是每个视频都支持了多种分辨率。  
每个视频都有100+的主观评分，是1-5的值。还人工打了内容类型标签。  
官网上有下载链接不知道能不能下，YUV的格式2T、H.264 110G、VP9的只有20G。  
* LSVQ  
2020年还是UT Austin做的一个目前数量最大的VQA数据集，真的好有钱。包含39,000 个现实世界的失真视频和 117,000 个时空局部视频补丁（“v-patches”，TODO：这是什么？）。  
有 5.5M 人类感知质量注释。  
原论文Github上有下载工具，听说可能下不全。自己试了下还真是有问题，分了两部分，第一部分是给了colab脚本去下载，速度非常慢……第二部分还是box网盘填写信息密码发邮箱，但是收不到密码……Hugging Face上teo wu看传了一份，大概不到100G可以下。  
* MSU-VQMB  
就36个视频，不过都是比较高清的。  
总共收集了来自 10,800 个参与者的 766,362 个有效评价。  
* DIVIDE-3k  
2022年Weisi Lin老师团队考虑审美分数新建的数据集，3634个训练视频，909个验证视频。  
每个视频都有审美分数、技术分数和整体分数  
TODO  
### PCQA数据集  
WPC、SJTU-PQA、M-PCCD、IRPC  
SIAT-PCQD、PointXR，一般点云质量评估是固定观察距离的图片/可旋转视角去评估，这两个是支持6DoF移动观察的  
#### WPC  
Waterloo Point Cloud，滑铁卢大学发的一个比较大的彩色点云质量评估数据集，对应论文Perceptual Quality Assessment of Colored 3D Point Clouds（2021）  
  
# 一些研究的思路  
## 特征工程  
### 预处理（采样）  
在VQA任务的数据预处理中，采样是核心，可以降低处理数据开销、匹配模型输入size、提升算法性能  
#### 空域  
* crop  
分为随机、中心、指定点等等  
* resize  
一般是缩小的resize，这也是一种下采样，涉及到不同算法如最近邻、双线性、Area、Lanczos等等  
* 网格采样拼接  
FastVQA的Grid Mini-patch Sampling（GMS）方法，分网格后在网格内部随机采样然后再拼接成一块  
##### 整理一点IQA模型的经典处理  
23 TOPIQ 384x384输入，所以大图resize到448/384，小图就直接随机crop  
23 HyperIQA 一张图像取3+15？多个子图像输入  
22 MANIQA 224x224输入，一张图随机crop 20次取平均  
21 UNIQUE 384x384输入，大图resize到512x512，然后随机crop  
21 TReS 随机crop50x224x224输入  
20 KonIQ 适配数据集的512x384输入，做了resize 224x224版本性能不好  
19 DBCNN 最后有全局池化，支持任意输入，但还是resize到448/384  
17 NIMA-inceptionv2 resize到256，再随机crop 224  
重复crop验证 HyperIQA TReS MANIQA LIQE  
分层特征连接  
#### 时域  
* 随机采样  
* 均匀采样  
* 关键帧  
需要额外获得关键帧信息，场景变换一定有关键帧可能会处理的好一些  
* 平均分段后段内随机起始位置等间隔采样（可重叠）  
首先是经典TSN网络提出的分段clip后随机采样一帧方法，简单有效。后面又升级了下，在一个分段内随机起始位置按间隔连续采样n帧。  
这个算是比较通用的做法了，可以记为clip_len x frame_interval x num_clips，即每一段采几帧x间隔几帧采一帧x总共分几段。具体实现参考openmmlab的mmaction2中SampleFrames。注意是允许不同片段采样帧重叠的，例如30帧，clip_len=5，frame_interval=2，num_clips=3，做法不是分成不重叠10x3的三段，然后内部抽5帧，实际实现是(30-5x2+1)/3=7，得到分段起始点为[0,7,14]，然后每一段0-6随机偏移，例如变成[2,10,18]，接着顺着间隔2帧采连续5帧。  
* 平均分段后段内随机起始位置等间隔采样（不可重叠）  
感觉这个想法更直接一些，分段然后每个段里取自己的。这也是Faster-VQA的做法，说相当于时域的网格采样，还挺形象，但感觉实际和上面可重叠的做法没太大区别，减少一点采样数不是一样的吗？  
#### 进阶思路  
##### 多尺度  
空域中低分辨率输入拿全局信息，高分辨率原始输入获得局部纹理细节信息  
##### 时空数据融合  
例如TSM  
### 分析特征权重（ROI/显著性检测）  
模拟人类视觉更关注图片重点内容，显著的区域质量重点处理。ROI (Region of Interest, 感兴趣区域)  
可能需要结合ROI、显著性检测、边缘检测等方法  
最终的效果就是拿到了图像某一区域的权重信息（可能二值也可能多级），有了这个信息就可以做分层采样、分区域评分增加权重等等操作  
#### 时域重要片段检测  
和空域同理，时域上也会有重要的视频片段和不重要的片段。在视频理解等任务中可能比较明显，但是VQA任务中按说没有那么突出，可以作为一个考量。  
可能需要结合动作识别、场景识别等模型。  
如果拿到不同时域片段的权重信息，也可以时域分层采样、分片段设置评分权重等  
### 特征融合  
不同来源不同层次的特征搞在一起。手动搞得特征很复杂的时候就需要想办法怎么合在一起送入网络。  
#### 早期融合  
直接送入Transformer的话基本上就是想办法都搞成Token拼接在一起，或者加和在一起（如位置编码）  
#### 中期融合  
在中间层合并一些特征，这里能做的比较复杂，例如一些旁路模型去获取权重信息去对中间层特征做加权  
#### 后期融合  
这个也可以说是多模型集成学习，即多个模型都给出结果了再去得到一个最终结果。  
### 结合传统特征  
* 频域信息  
* 直方图  
## 模型魔改  
### CNN分层特征  
### 注意力机制Block  
### 新模型  
#### Mamba  
#### RMKV  
#### 仿HVS模型  
### 多模型融合  
#### 集成学习  
多种各有优势的网络模型合在一起，利用各自模型的优势场景分开处理不同的输入，获得的结果可以再通过简单的投票选一个/加权平均、复杂的MLP网络等方法，得到最终结果  
模型可以分开训练。但是推理、部署的复杂度还是会提高。  
#### 混合专家MoE  
通过门控机制来根据情况选择最适合的模型  
动态选择模型，训练会比较难做  
### 压缩模型  
#### 知识蒸馏  
小模型直接学习大模型的预测结果，实现更低的开销获得类似的性能  
#### 量化  
混合精度训练AMP，已经做了。据说CV任务中bf16可以了，再小会有问题  
后量化  
#### 裁剪  
CNN好做一些，看权重低的连接、filter、通道就直接移除了。Transformer中可能也类似，多头去点头、减少层级、去除权重低的连接。  
放到Mamba里面感觉不敢动了，原理也都理不太清……  
## 数据集问题  
### 主观评分  
主观评分很难搞，找人评图像/视频建立数据集还是成本很高的事情，而且还需要按标准才能得到比较靠谱的主观评分  
#### LLM  
结合LLM多模态输入图像给出文本评价的能力，进行主观评分  
### OU-BIQA  
无主观评分训练数据做NR IQA  
#### Learn to Rank  
获取绝对评分困难，但是比较两幅图像优劣相对而言简单，有这个信息也可以间接进行评分  
### 数据增强  
感性的结论是VQA任务不应该做数据增强。经典的一些方法例如加噪声、变颜色、Mixup等方法应该都是不能用的，对画面质量会有影响。  
一些空间变换如翻转、裁剪、resize理论上可能有效果。但负面影响未知。  
视频画质增强、生成模型绘制等方式明显也会改变质量评分。  
那有没有能用的？用了之后效果如何？其实是值得分析的问题。  
#### JND  
最小可察觉误差JND，把来自人眼视觉系统特性&心理效应的视觉冗余量化出来，可以指导编码和质量评估。通过JND来生成一些评分接近的训练数据做增强  
#### 量化相对得分  
分析resize、crop、sharpen等一些图像处理操作对图像质量评分的相对影响，在对原数据进行一些操作进行数据增强时，可以根据相对得分计算出一些假的分值，或许可以有用  
这个是不是和zero-shot相关，通过数据的关系去推测标签的关系？  
### 小数据集/零样本学习  
#### 大数据预训练+小数据集微调  
大数据集LSVQ上预训练的再放到小数据集上微调  
#### 相似任务预训练+小数据集微调  
用数据量多的视频理解等任务预训练的模型作为底层模型，提取出公共的特征  
#### 多任务学习  
让模型同时学习多个相似的任务，可能能更好地提取出公共特征，损失函数就是多个任务合起来，例如同时跑目标检测、语义分割、姿态估计  
#### 动态调整模型  
VQA任务的主观性很强，如果不同用户有不同的评判标准（尤其是审美方面），已训练好的数据，能否实现用户自己个性化微调的效果？  
在线学习？增量学习？  
#### 自监督学习  
跑不需要标签但是能获得和目标任务相关特征的一些任务来预训练模型  
##### 对比学习  
构造正负样本对，正样本对是原数据中同一个样本数据增强得到的，负样本对是原数据中不同样本（数据增强得到的），然后模型提取样本特征后用于计算相似度，目标是正样本对相似度高，负样本对相似度低，这样得到的模型也能提取有效特征  
##### MAE  
mask掉原数据的一部分让模型补全  
  
## 参考其他任务  
### 和视频理解任务关联  
视频质量评估任务和视频理解任务相对而言是相似程度比较大的，都需要对视频空域时域的信息进行分析。而相比之下，视频理解任务是一个更加热门的研究领域，实际中很多VQA任务模型也会用到从视频理解任务迁移过来的东西  
### VQA和IQA关联  
VQA相当于IQA加时域，在空域很多处理是相通的  
  
# 代码  
## IQA  
### IQA-PyTorch  
非常牛的一个IQA任务开源库，NTU Chaofeng Chen大神做的，搭好了一个通用框架，基本集成了主流的各个IQA模型和数据集，统一了训练、测试方法  
#### 代码使用  
1. 安装  
应该还是比较好装的，感觉不一定要安装，装了可以用命令行工具pyiqa方便一些启动固定的IQA任务，并且随处import pyiqa，不过直接跑也可以  
```  
git clone https://github.com/chaofengc/IQA-PyTorch.git  
cd IQA-PyTorch  
pip install -r requirements.txt  
python setup.py develop  
```  
2. 准备数据集  
常用IQA数据集都上传到Hugging Face了，真是一件大好事  
https://huggingface.co/datasets/chaofengc/IQA-Toolbox-Datasets  
使用的时候下载数据集对应压缩包，推荐在代码目录创建datasets目录，然后解压到这里，可以匹配配置  
除了数据集本身，所有数据集的标签文件和数据集划分设置，都打包到了一个meta_info.tgz文件，这个也要一起下载了解压到datasets目录  
除此之外还可以关注下DataLoader的实现，基类定义在  
pyiqa/data/base_iqa_dataset.py中，派生出通用的NR数据集DataLoader定义在pyiqa/data/general_nr_dataset.py中，然后一些数据集还需要有特殊的处理的要再自己派生实现DataLoader类  
3. 快速跑推理  
-m指定模型，-i指定输入图像  
```  
python inference_iqa.py -m brisque -i ./ResultsCalibra/dist_dir/I03.bmp  
```  
4. 跑训练  
训练的参数配置都是yaml配置文件中写好的，不同模型和数据集的配置还可能有一些细微区别，反正解析配置的代码也是可以派生类里自己处理的  
```  
# 普通数据集训练使用train.py  
python pyiqa/train.py -opt options/train/DBCNN/train_DBCNN.yml  
  
# 需要n-fold交叉验证训练的用train_nsplits.py，只是在train.py的基础上加了个循环不同数据集划分的操作  
python pyiqa/train_nsplits.py -opt options/train/DBCNN/train_DBCNN.yml  
```  
#### 新增模型  
这里代码做的挺牛逼的，要兼容不同的数据集以及不同模型结构，又要尽可能复用代码和配置项避免做成不同模型和数据集的代码集合，就是要写好的基类留好接口可能定制化扩展  
重点看好从网络结构、model、训练配置中各个部分已经有的功能，然后根据待新增模型的输入输出、预处理、训练方法等等方便去自定义开发  
##### 网络结构arch  
网络结构定义在pyiqa/archs目录下，可以任意新增网络结构，需要  
```  
from pyiqa.utils.registry import ARCH_REGISTRY  
```  
引入注册方法，然后把新增的网络结构类注册  
设计到预训练模型加载，有通用的配置项pretrained_model_path，如果要加载预训练模型的话，可以在init函数中判断配置自己进行处理  
##### 新增model  
model定义在pyiqa/models目录下，其实已经有比较好的一个基类GeneralIQAModel，基本把训练、测试时各种参数初始化、LOSS设置、模型输出都封装好了  
如果确实有一些特别的训练流程处理就单独写派生类修改，例如DBCNN增加了一个学习率降低但整个网络不冻结的微调阶段、WaDIQaM模型两部分用不同的优化器等等  
##### 新增训练配置option  
训练配置在options/train目录下，（test目录下还有超分模型配置，作者似乎想把超分的也整合了，看了下实际没有做），配置项支持的还是比较全面的  
分了基础配置、数据集配置、网络结构配置、训练配置等，具体配置的实现要结合代码去分析，有一些配置项并不是通用的，如果需要自定义一些配置也很好处理  
新建模型的时候综合相似的各个模型配置参考，做适合自己模型的配置  
  
  
# 论文整理  
## IQA  
### （17.1.13德国Fraunhofer研究所 DeepIQA/WaDIQaM）Deep Neural Networks for No-Reference and Full-Reference Image Quality Assessment  
非常经典的早期DL IQA模型，结构是很工整的，分开做了FR和NR的，单独看NR的吧  
特征提取部分就是（2层CNN+1层2x2最大池化）x5，图像拆分为32x32的小补丁输入，所以经过卷积之后正好size为1，通道升到512维。  
回归Head是分了预测和权重两路，每个小补丁预测分数，并预测补丁权重，最后加权平均获得整个图像的分数。  
跑分在FR数据集上还可以，但是在NR数据集上比较差了。  
不限制输入大小，训练的时候是每张图像随机crop 32个32x32的小补丁，测试的时候是整张图像不重叠的切补丁输入。  
  
### （19.7.5武大DBCNN）Blind Image Quality Assessment Using A Deep Bilinear CNN  
效果很不错的CNN模型  
网络结构分了两路，一路是只保留Backbone的S-CNN合成失真分类任务预训练模型，一路是只保留Backbone的VGG-16 ImageNet分类预训练模型，相当于合并了两个预训练CV模型。  
两路的合并方式是bilinear pooling，得到512x128的特征图。  
回归Head就是一层线性回归层，512x128->1  
S-CNN和VGG-16本身应该都是限制224x224的，但是合在一起做了池化应该是不限制输入大小的，所以实际训练的时候就是随机crop到448/384，验证好像就是原图  
  
### （19.12.20德州大学）From Patches to Pictures (PaQ-2-PiQ): Mapping the Perceptual Space of Picture Quality  
做了一个较大的in-the-wild IQA数据集（应该是FLIVE），并且每张图像还带了3个不同大小patch（20% 30% 40%）的MOS，从而可以分析下局部和整体的质量关系，答案是比较相关（算是随机crop的理论基础了）  
做了个模型就是ResNet-18改改，因为有了patch评分的信息，所以引入了ROIPool加权，把patch质量评估结果和全局质量评估结果一起考虑得到最后得分，会比光输入图片高一些  
感觉这也不太算是数据增强的技术，patch也是做了主观评分实验的……  
  
### （20.3.7西工大 HyperIQA）Blindly Assess Image Quality in the Wild Guided by a Self-Adaptive Hyper Network  
TODO：思路没太看明白，大致是把ResNet的高层语义特征和底层纹理特征分开处理了  
ResNet之后高层语义特征分了一路输入啥内容理解网络，分层特征输入MLP。然后内容理解网络的分层输出又输入MLP，反正最后出的结果抽象理解下就是综合了高层和底层的特征，把CNN用的比较充分了  
输入是一张图像随机crop 25张224x224然后结果取平均  
  
### （20.4.11西安电子科大）MetaIQA Deep Meta-learning for No-Reference Image Quality Assessment  
好像是在合成数据集上训练然后去跑真实失真数据集，普通CNN模型  
合成失真数据集上还可以，但是真实失真数据集上指标挺差，感觉参考价值有限  
  
### （20.12.30挪威研究中心 TIQAorTRIQ）Transformer for Image Quality Assessment  
应该就是比较早的Transformer用法  
ResetNet提取出的特征图输入ViT  
回归Head两层MLP  
没做resize和crop，说没限制输入size，把token限制尽量搞大了  
  
### （21.2.23上交 UNIQUE）Uncertainty-Aware Blind Image Quality Assessment  
TODO：这里原理没太看明白，设计了一个稍有点复杂的训练方法，可以把多个IQA数据集放在一起训练  
网络就是基于Resnet-34  
回归头是一个512x512->2的线性回归层  
训练阶段输入resize到一边512，然后随机crop到384  
  
### （21.8.12谷歌）MUSIQ Multi-scale Image Quality Transformer  
指出了CNN-based模型存在输入图像size限制问题，说明了一般的解决方法：  
* resize&crop -> 引入失真  
* 多crop集合 -> 增加计算成本  
* 池化到固定形状  
* 提取固定大小特征  
  
虽然是文章中提出要解决的问题，但其实上面的方法还挺值得借鉴的。Transformer按说也是ViT那样固定224的输入，但是有办法做一些拓展。  
网络直接用的Transformer，但是在Embedding部分做了很多文章，可以支持任意宽高比的全尺寸图像直接输入  
Embedding部分引入了三个组件，一是多尺度Patch Embedding；二是基于哈希的二维空间Embedding；三是可学习的scale Embedding。TODO: 细节可以研究下，最后的效果是输入的图像补丁还加入了多尺度的信息  
回归头就是一个线性回归层  
任意原始尺寸图像输入，有一种优化是resize到384和224加两路输入，有一点提升但是基本可以忽略  
  
### （21.8.16CMU TReS）No-reference image quality assessment via transformers, relative ranking, and self-consistency  
https://github.com/isalirezag/TReS  
CNN+Transformer用的比较好的  
ResNet50分4层特征图，池化统一大小之后连在一起输入Transformer  
回归Head  
输入是一张图像随机crop 50张224x224然后结果取平均，有点夸张  
  
### （21.10.25德州大学 CONTRIQUE）Image Quality Assessment using Contrastive Learning  
预训练阶段跑自监督学习——对比学习，这里做的对比学习任务还是需要一点信息的，FR数据集上学习合成失真的知识，D种失真x(L+1)种失真级别可以分为D(L+1)类，然后跑对比学习拉近同类推远异类。NR数据集上学习真实失真的知识，这个理论上不那么好做，每张图像都是独立的，所以做个下采样就有同类了，和FR数据混在一起做训练。  
预训练阶段获得的训练好的ResNet50，冻结住，然后加一个简单的线性回归头，这个线性回归头还是在目标数据集上训练的，想到于用对比学习获得的Backbone来提取特征。指标看起来还行，应该是FR的合成数据集上表现好，NR数据集上还是一般的。  
  
### （22.4.29清华）MANIQA Multi-dimension Attention Network for No-Reference Image Quality Assessment  
非常经典的Transformer IQA模型，指标爆杀之前所有模型，把注意力机制用的很好，值得学习  
网络结构是直接ViT，输出结果又拼成图像之后过了卷积和Swin Transformer组成的Block，最后过一个两路的双层MLP，分别获得patch的分数和权重，最后加权求和得到整个图像的评分  
输入是训练阶段一张图像随机crop一张224x224（反正多轮训练），测试阶段一张图像随机crop 20张224x224然后结果取平均  
  
### （23.3上交BIQA模型LIQE）Blind Image Quality Assessment via Vision-Language Correspondence A Multitask Learning Perspective  
把多模态大模型CLIP用于IQA任务，效果还可以，但是看指标还是弱于MANQIA的  
在IQA任务中文本-图像对中的文本内容包含了场景、失真、质量评分信息，例如“xx场景图像，有xx失真，质量xx”。  
训练时做了多任务训练，就是同一个模型，可以跑场景分类任务、失真分类任务、IQA任务三种，合起来算一个Loss去训练  
文本encoder用的是GPT-2，图像encoder用的是ViT-B/32  
好像这里没有啥回归头，文本分支出的Embedding和图像分支出的Embedding得到余弦相似度就直接作为输出了，使用不同的的view、sum和argmax操作得到三个任务各自的预测结果，还挺神奇的  
图像等间隔crop 15张224x224然后结果取平均  
  
### （23.4.2德州大学）Re-IQA Unsupervised Learning for Image Quality Assessment in the Wild  
MoE的模型，组合了两个无监督学习训练出来的模型，用的是对比学习？的方法  
一个模型用来提取高层表示，理解图像内容，做法是用ImageNet上预训练的Resnet50再去怎么学？  
一个模型用来提取低层表示，感知图像质量，做法还没看懂……  
TODO  
  
### （23.8.6NTU）TOPIQ A Top-down Approach from Semantics to  Distortions for Image Quality Assessment  
TODO：模型稍有点复杂没太研究明白  
核心的一个思路是考虑高层特征中包含的语义信息，然后用高层特征指导自顶向下指导底层特征，特征连线有点复杂，主要的改动都是在一些特征融合模块上，引入了gated local pooling block（GLP），self-attention block（SA），cross-scale attention block（CSA）三种模块  
分了FR和NR两种模式，FR就要多融合一些参考图像的信息，看跑分还是相当高的  
回归Head是一个三层的MLP，感觉不算很重要了，前面还有注意力模块，整体已经很复杂了  
输入应该是任意size的，毕竟有GLP池化统一特征图大小，不过训练的时候针对不同数据集还是做了随机crop到384或者224，原图很大也会resize到448/384-416  
  
### （24.5.29港城大 Compare2Score）Adaptive Image Quality Assessment via Teaching Large Multimodal Model to Compare  
引入LLM进行比较评分（对比两张图片给出差、较差、相似、较优、优的相对评级），也算是Q-Align模型引入LLM做主观评分的进一步拓展，可以绕开不同数据集之间MOS评分不一致没法混用的问题  
TODO：分析下这里对比评分的用法  
  
### （24.5.29港城大 MDFS）Opinion-Unaware Blind Image Quality Assessment using Multi-Scale Deep Feature Statistics  
这里讨论的是比较小众的OU-BIQA（BIQA就是NR IQA）任务，也就是没有带主观评分标签的训练数据，依然要做IQA，听起来就是没什么依据很难做。  
做法挺有意思的，偏统计的方法。先是用预训练的CNN模型提取分层的图像特征（所谓的MDFS多尺度深度特征提取是金字塔形式下采样，前一层特征做下采样和后一层一致，最后统一特征图大小连接起来），通过统计方法把特征图转换为一个多维高斯分布，也就是实现了图像->高斯分布的映射。  
评分的时候一方面搞了个高质量图像的数据集作为高分的基准，算出来这个数据集平均的高斯分布，然后测试图像也搞成高斯分布，计算两个高斯分布之间的距离得到评分。  
想法很有意思，其实算是造出了FR IQA了。NR IQA中没有对应的参考图，随便找一些高质量图像做参考图？直接比较测试图和参考图的相似性并不好做，毕竟内容都不一样，但是过DNN获得的特征图再转统计学高斯分布表示，再计算相似性就有那么点道理了，可以认为减少了具体的内容信息，而是有一些内容无关的统计量表示，就能通过相似性发现图像退化程度  
  
### （24.8pub NTU CMKernel）Continual Learning of Blind Image Quality Assessment with Channel Modulation Kernel  
TODO： 用了个通道调制核，没看明白是啥  
  
## VQA  
有很多早期的模型现在看来指标都比较差了，所以没有往前看很多，可能会漏掉一些有意义的idea，有机会再看吧。整体看近年的论文大都是在采样方法上做文章，还是有很多相通之处的。  
### 综述  
#### （24.2.5上交 超强综述）Perceptual Video Quality Assessment: A Survey  
这篇综述覆盖的内容非常全面，主观评分方法+数据集+通用VQA方法+特定应用VQA方法+指标+展望  
整理了大量数据集和模型，尤其是除了通用VQA之外，还覆盖了很多新的VQA方向（特定应用），能有一些启发，可能针对一些特定的应用去设计有侧重点的模型，不一定一直卷通用VQA。例如视频压缩、流媒体、3D视频、VR视频、高帧率视频、音频+视频、HDR/WCG视频、游戏视频……  
  
对于做的比较多的通用NR VQA部分，做了分类，这里挺不好分类的，只能大致分下，我觉得也挺有思路  
分为知识驱动（旧方法手动提一些特征）、和数据驱动（基于预训练模型和基于端到端训练、基于自监督学习）  
其中专门点出了大量模型是直接基于预训练模型的，这一点也是挺重要的，即使不从头训练模型，很多预训练模型可以直接拿来用，然后最后训练回归器就可以，但似乎这并不是优于从头训练的方式，毕竟有LSVQ数据集可以从头训练可能干不过，但看做的一些比较新的文章指标还不错  
自监督学习方法能够缓解主观评分数据不足的问题，应该是很有意义的，目前看论文有一些，要研究一下，看看和数据增强的关联  
  
### （18.7.31深大 TLVQM）Two-Level Approach for No-Reference Consumer Video Quality Assessment  
我觉得算是传统人工特征的一个集大成模型了，用了75个传统特征，什么对比度、运动矢量都上了，最后KoNViD-1k上跑出SROCC 0.78，是不错的成绩了，但也基本说明做到这种程度也依然被深度学习方法吊打。  
  
### （19.10.5北大 VSFA）  
问题：VQA有时间滞后效应，人会记住历史的低质量帧然后降低对后续帧的评分，需要考虑长期依赖关系  
方法：先使用CNN提取每一帧的内容特征，每一帧的特征按时间顺序输入到GRU中，使用GRU来解决对历史帧记忆的问题。最终的评分还经过了各个时刻GRU输出结果做池化  
想法：这个处理时间滞后效应问题的想法挺有意思的，给出了用RNN模型的理由。但是从现在的视角看这个问题似乎也没有那么重要，毕竟这里早期模型指标跑出来挺一般的，有点硬加了下GRU的感觉，现在应该很多ViT的模型有多帧信息都是可以处理一点长期依赖的  
### （20.11.27德州大学）Patch-VQ 'Patching Up' the Video Quality Problem  
就是带了LSVQ数据集的经典早期文章。同时做了个PVQ模型，应该说是稍微粗糙的一个经典baseline，基本上后续模型对比都会看到。  
文章中有个很重要的结论，就是视频global和local的主观评分是比较相关的，算是空域&时域采样的理论基础了。具体分析方式是做数据集的时候，搞了3w+的原始视频，又使用非常初级的三种空域时域采样方法得到了3倍的patch，分别MOS评分判断相关性。采样方法有：空域采样（时域不采样，空域随机crop 16%面积，SRCC=0.69）；时域采样（空域不采样，时间随机连续的40%，SRCC=0.77）；时空域采样（前面的空域采样+时域采样一起做，SRCC=0.67）  
  
一 关于数据集  
强调了UGC视频有各种时空域的短暂失真，包括丢帧、焦点变化、传输故障，而现有UGC数据集太小覆盖不了多少情况。所以做了一个目前最大的UGC数据集。  
二 关于PVQ模型  
这个模型的结构看起来略显复杂，但实际上和后续的SimpleVQA基本是一致的，只是当时还没有那么清晰的思路说一路输入2D帧取空域内容&纹理特征，一路输入低清3D取时域动作特征。  
具体到细节上是用PaQ2PiQ网络提取2D的视频特征，ResNet3D网络提取3D的视频特征，然后做了ROI和SOI的池化（这个效果如何值得分析下），最后用一个InceptionTime时间序列回归模型来给回归得分。  
  
想法：SimpleVQA对比这个，差不多就是升级了2D和3D视频特征提取网络的Backbone，改为了，池化和回归好像都做的更加简单一些，我觉得是一个非常好的结构设计。  
### （21.6.22港城大 GST-VQA） Learning Generalized Spatial-Temporal Deep Feature Representation for No-Reference Video Quality Assessment  
这个好像后面模型对比的少一些，主要的思路是想提取多尺度特征，从而可以适配不同分辨率，不同时长的视频输入。  
我觉得比较特别的点是网络结构设计的好像稍有点复杂，我有点没太看明白，大概是VGG16->Transformer->MLP->GRU，认为这样子有空域的多尺度特征，然后对于GRU输出的多帧时域结果，训练阶段还做了高斯正则化，测试时还做了金字塔特征聚合，可能对短期长期的时域影响处理的更好。  
因为最后指标跑出来一般般，这个模型不细分析了，往多尺度的思路去思考还是可以的。  
  
### （21.8.19上交 BVQA）Blindly Assess Quality of In-the-Wild Videos via Quality-Aware Pre-Training and Motion Perception  
SimpleVQA的前身，大体思路是相似的。  
还得到了结论是在IQA数据集和动作识别数据集上预训练的模型迁移到VQA任务中是有效的。  
### （21.8广州大学）StarVQA Space-Time Attention for Video Quality Assessment  
算比较早用纯ViT的模型，创新性上倒是没啥特别的地方。看是直接随机crop224x224然后输入ViT了，指标跑出来在大数据集LSVQ上挺好的，小数据集上就比较差，毕竟ViT。  
如果需要对比ViT Backbone的话可以看下这个代码。  
### （21.10挪威研究中心 LSCT-PHIQ）Long Short-term Convolutional Transformer for No-Reference VQA  
用了两个看起来结构很复杂的网络，跑了KoNViD和Youtube UGC指标挺不错的。TODO  
先是PHIQNet，是一种考虑了多尺度的CNN，提取出考虑多尺度的特征。  
然后是Long short-term convolutional transformer，长短期卷积Transfomer？  
  
### （22.5.7福州大学 STFEE）Deep quality assessment of compressed videos A subjective and objective study  
这里预处理阶段就做了显著性检测，是一种加ROI的方法，然后送入I3D提取特征，最后transformer回归  
  
### （22.6.6 NTU）FAST-VQA: Efficient End-to-end Video Quality Assessment with Fragment Sampling  
问题：主要解决了VQA中高清视频数据量过大，计算开销高的问题  
方法：  
* 网格化补丁采样  
把原始视频帧按网格分块，然后每一块里原始分辨率采样一个小补丁，不同帧都在相同位置采样这个小补丁。——这样就采样获取了最细节局部纹理信息  
把所有网格的小补丁拼接在一起组成一个“大补丁”作为输入。——这样就有了个很抽象的全局语义信息（直观感觉有点弱，这都成块了啥也看不出来）  
* FANet网络  
backbone用的是4层注意力层的Swin-T网络，但还要做一点调整。主要问题是大补丁毕竟是硬拼起来，小补丁内Intra-patch像素关联强，小补丁之间cross-patch像素关联弱。所以，一是做注意力计算的时候要有所区分，加一个偏差值隔开不同小补丁；二是不能先池化再非线性回归（这是激活函数吗？）会把小补丁混了，顺序改成先非线性回归再池化。  
  
想法：这样很小的采样又硬拼在一起，相比更早的方法直观感觉是增加了细节纹理的权重、削弱了全局语义的权重，也能取得很好的效果，这一点是挺令人惊讶的，应该说明了目前的质量评分基本上还是以局部细节纹理为主吧。  
但是由于随机采样引入了评分结果不稳定的问题，只有在整体数据集上统计表现不错，但对于具体单个视频误差较大。  
  
22.10的Faster-VQA又在时域采样上搞了点操作说是时域的网格采样，但是个人觉得没什么意思，和普通的时域采样没明显区别。  
  
### （22.6.20 NTU） DisCoVQA: Temporal Distortion-Content Transformers for Video Quality Assessment  
问题： 时域失真（shaking、 flicker、abrupt scene transitions）如果仅仅是固定时域采样的话难检测，需要重点关注时域失真的VQA模型  
方法：  
通过一个Transfomer提取时域失真（TODO：怎么实现的？是每一帧都输入吗）并获取每一帧的质量评分，这个Transfomer的输出再输入到另一个Transfomer进行内容的分析给出每一帧的质量权重，加权得到最终评分。  
这样的做法明显就是给了时域质量更多的权重，看起来指标一般吧，可能还是需要时域失真比较明显的数据集才能有明显优势。  
  
### （22.6.29德州大学）CONVIQT Contrastive Video Quality Estimator  
TODO：引入对比学习的IQA模型CONTRIQUE的VQA版本  
  
### （22.7.8NTU TPQI）Exploring the Effectiveness of Video Perceptual Representation in Blind Video Quality Assessment  
问题：   
还是关注时域失真，视频的时域质量和人类视觉感知之间的关系不明确  
做的是小众的Completely BVQA，也就是无标签训练数据的OU-BVQA  
方法：  
发现有个HVS域变换的方法，可以把一段视频帧转换为变换域的一个直线轨迹，而一旦有失真轨迹就会乱，变成混乱的弯曲轨迹。所以提出可以用一个TPQI指标，度量HVS变换域轨迹图的形态（straightness平直度 and compactness紧凑度），从而反应出视频时域质量。这种比较客观的指标就不需要训练模型去调参提特征做回归，可以跑OU-BVQA任务  
具体的变换方法有两个  
* lateral geniculate nucleus (LGN)  
* primary visual area (V1)  
  
### （22.10.9广州大学 HVS-5M）HVS Revisited: A Comprehensive Video Quality Assessment Framework  
拼了5个模块提取5种特征输出最后的总分，指标也还不错，这几个特征可以关注下，也是比较直接的加人工特征的方式。TODO  
用的Motion Perception、Temporal hysteresis、Content dependency、Visual saliency、Edge masking  
  
### （22.10.20上交 SimpleVQA）A DeepLearning based No-reference Quality Assessment Model for UGCVideos  
和FastVQA一样也是目前VQA的经典模型，值得学习。  
核心点是在预处理采样的时候，分了两个路径，一个是高清少帧数的一路看细节纹理，一个是低清高帧数的一路看动作内容。  
空域采样就是采少量分辨率较高的帧520x520再中心crop到448x448，送入ResNet50提取空域特征,。  
时域采样是这个片段全部帧但是resize到低的分辨率224x224，送入SlowFast提取时域信息（应该是动作识别一类的结果）。最终两部分特征再合起来经过一个MLP得到最终评分。  
  
### （22.11.9 NTU DOVER）Exploring Video Quality Assessment on User Generated Contents from Aesthetic And Technical Perspectives  
问题：之前的VQA只关注了技术角度的质量评分  
方法：  
* 做了DIVIDE-3k数据集  
每个视频标签是技术评分+审美评分+整体评分  
* 提出了DOVER视频质量评估模型  
就是分了两路分别计算技术评分和审美评分  
TODO：DOVER具体网络设计，我理解技术分应该还是类似Fast-VQA的处理，审美分主要依赖全局语义可以做下采样，但是提了个Cross-scale Regularization没看懂是什么，不同尺度下采样之间比较吗？  
  
### （23.1.3谷歌）MRET Multi-resolution transformer for video quality assessment  
TODO  
  
### （23.4 NTIRE VQA竞赛）NTIRE 2023 QA of Video Enhancement Challenge  
淘宝团队基于SimpleVQA的模型拿了冠军，在数据增强上做了比较多。TODO：需要借鉴下  
### （23.4.19上交）MD-VQA Multi-Dimensional Quality Assessment for UGC Live Videos  
TODO  
  
### （23.5.22 NTU MaxVQA）Towards Explainable In-the-Wild Video Quality Assessment: A Database and a Language-Prompted Approach  
基于DOVER时搞的DIVIDE-3k数据集，又扩展了评分的维度，本来只有审美评分+技术评分+整体分，又对数据集里4543个视频收集了200w+的文本评价，把这些评价转换成了13个维度的评分。这个数据集确实是个极大的亮点，借助文本语义的信息来解决原本视频质量只有技术评分的问题。  
提出了MaxVQA，这里的Max指的是多维度。  
TODO：网络设计需要再看下，技术分还是用了Fast-VQA，其他语义评分的处理用了做多模态的CLIP  
### （23.7快手 VQT） Capturing Co-existing Distortions in  User-Generated Content for No-reference Video  Quality Assessment  
TODO  
  
### （23.8.1清华）Ada-dqa Adaptive diverse quality-aware feature acquisition for video quality assessment  
应该是预训练模型使用的集大成之作  
拼了7个经典预训练模型，最后回归，还做了蒸馏到一个Video Swin Transformer模型，分数很不错  
  
### （23.12.28 NTU）Q-ALIGN: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels  
使用多模态大模型，输入是图像和文字问题（评分如何？），输出是5个主观评分级别。训练后可以实现利用大模型来做主观评分，并且同时支持图像质量、图像美学、视频质量的评分。  
  
### （24.2.20快手 KSVQE）KVQ:KwaiVideo Quality Assessment for Short-form Videos  
应该是目前的SOTA，在FastVQA网格采样的基础上，又加了两个旁路网络，一个是获得不同分块的权重（似乎还有内容识别），一个是检查一些形变失真（这个听起来没啥意思）。  
TODO：这个要认真分析下。  
  
### （24.2.29港城大）Modular Blind Video Quality Assessment  
CVPR 2024  
还是个集成3个模型的VQA模型～  
目前的VQA模型比较难处理的一个问题就是整个视频输入进去太大了没办法处理，无论原视频多大，固定时域采样只取8/16/32帧，空域采样每一帧裁剪/网格采样到224x224，这样就会丢时域和空域的信息，一些失真就发现不了（但从公开数据集上的跑分看，采样很少依然效果不错）。按说视频分辨率帧率越高采样损失就越多，所以很多模型就会提一些方法来找补。这里是：  
1. 还是用个普通VQA模型，时域空域都采样（文中的基础质量预测器）  
2. 加一个空域不采样、时域采样的VQA模型，也不是输入全图，是算了全图的拉普拉斯金字塔做特征（文中的空间整流器）  
3. 加一个时域不采样、空域采样的的VQA模型，可以多输入一些连续帧了（文中的时间整流器）  
最后拿2、3模型的输出矫正1模型的输出，思路挺清楚的，理论上能发现一些别的模型发现不了的问题，就是在大部分公开数据集上刷分看提升不大，在4K、120Hz这种高分辨率高帧率的数据集上效果好，但复杂度估计也挺高  
  
### （23.4.12NTU PTQE）Blind Video Quality Prediction by Uncovering Human Video Perceptual Representation  
22年TPQI的进一步扩展  
TODO：  
  
### （24.4.17快手组织竞赛论文）NTIRE 2024 ChallengeonShort-form UGC Video Quality Assessment: Methods and Results  
可以看到很多魔改模型的思路去借鉴。因为卷的是准确率，所以基本上都是多模型在一通拼，魔改的会很复杂。  
第一名是上交SimpleVQA作者和NTU FastVQA作者的联队，这有点欺负人了，直接把SimpleVQA和FastVQA的结果组合了，还附加了Q-Align和LIQE两个模型的结果一起，4个拼在一块很强。  
但是整体看下来没有什么新的东西，毕竟是刷分的比赛，基本都是流行的这几个主流模型去一通合并，不过也可以看出来FastVQA（含Dover）、SimpleVQA、Q-Align等几个模型是经过检验的模型。  
### （24.5快手 PTM-VQA） PTM-VQA: Efficient Video Quality Assessment Leverage Diverse PreTrained Models from the Wild  
TODO  

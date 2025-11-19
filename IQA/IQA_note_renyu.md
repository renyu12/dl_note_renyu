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
为什么要让机器和人打分差不多？因为在机器做一些图像&视频相关的优化任务时（例如压缩、增强、生成），需要有个量化的评价指标，乙方（机器）做事情不可能一遍一遍去问甲方（人）的标准，而标准不一就会导致乙方做的甲方不满意又打回重做，所以尽量一开始就统一标准。具体来说有以下一些场景需要VQA/IQA  
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
#### FR-IQA数据集  
比较经典的5个常用的有参考IQA数据集列在表里  
| 数据集 | 参考图像数 | 失真图像数 | 失真类型数 | 测试人员数 |  
| --- | --- | --- | --- | --- |  
| KADID-10k | 81 | 10125 | 25 |  |  
| TID 2013 | 25 | 3000 | 24 | 971 |  
| TID 2008 | 25 | 1700 | 17 | 838 |  
| CSIQ | 30 | 866 | 6 | 35 |  
| LIVE | 29 | 779 | 5 | 161 |  
  
注意FR数据集在划分训练集、测试集的时候，需要根据参考图像分组，例如TID2013 20张训练图像、5张测试图像（分验证集也是5张），避免模型学习到的是特定的内容特征而非泛化的质量特征  
记录详情便于预处理：  
* LIVE  
0-100区间（有越界，实际-2-107），低分代表高质量，分布较为均匀  
* CSIQ  
0-1区间（有越界，实际0-1.056），低分代表高质量，低质量样本更多  
* TID2013  
0-5区间（有越界，实际0.24-6.75），高分代表高质量，高质量样本更多  
都是512\*384的BMP文件。3000张=25张图片\*24种失真类型\*5个失真等级  
* KADID-10k  
0-5区间（无越界），高分代表高质量，分布较为均匀，非常好的数据集  
  
#### NR-IQA数据集  
无参数据集有  
* LIVE-Challenge  
2016年的一个in the wild数据集，1162张照片，8100人标注，35万份评分，MOS分数[3.42, 92.43]，高分代表高质量，高质量样本更多  
* KonIQ-10K  
2020年的一个in the wild数据集，10073张照片，1459人标注，120万份评分，MOS分数[3.91, 88.39]，高分代表高质量，高质量样本更多  
  
做美学评分专门有个很大的AIA数据集  
* AIA (Aesthetic Visual Analysis)  
论文是2012年发布的，数据集应该更早。255508张照片，每张图投票数78-549，平均210票，MOS分数1-10共十个级别  
每一个图像的分数都是给的1-10分值的分布。还有语义标签、摄影风格标签  
分析了分数分布发现基本都可以用高斯分布近似（个别新锐风格图片方差大）  
  
还有一些时间久的IVC、Toyama、A57、WIQ数据量比较小了  
TODO: 也有一些新的可能还没有完全普及作为通用标准，需要研究下，例如PieAPP, PIPAL, FLIVE, SPAQ  
  
### VQA数据集  
很多视频数据集的视频来源都是大的公开数据集YFCC100M（有793436个视频），只是没有标记。不同的VQA根据需求从中采样一部分视频去标注MOS  
有个VQEG视频质量专家组的组织整理了一些工作，可以关注下https://www.vqeg.org/vqeg-home/  
* CVD2014  
2014年赫尔辛基大学做的，算偏早期的数据集了。  
78 个不同的相机（手机、小型相机、摄像机、单反相机）拍摄的来自五个不同场景的 234 个视频组成  
主观评分这里好像分了组做实验，这里没看明白，可能不太容易直接拿来用。有6G、16G、23G三组数据。  
* LIVE-Qualcomm  
2017年UT Austin做的，8 种不同移动设备拍的208个视频，54个场景，一个视频15s，都是1080p的，模拟了六种常见的拍摄失真类别。  
每个视频都由 39 个不同的受试者进行评估  
* KoNViD-1k  
2018年，1200个视频，一个8s  
质量评分是1-5五个等级。  
大概2.8G，非常小，一个视频就1M左右。  
首先从YFCC100M数据集中粗筛出可用的14w+的视频，然后评估了视频的6个指标（模糊、色彩、对比度、空间、时间、NIQE质量），采样的时候重视多样性覆盖指标各个范围的视频，“fair-sampling”了10000个视频，然后从中又随机选了1200个。所以这精选出来的应该是很能代表YFCC100M数据集的。  
众包获取评分，通过检查评分偏离标准值+-1就是不准确，准确率低于70%的测评者被排除，每个人评550个视频，64个国家642人参与评分。  
视频是静音播放的。  
* LIVE-VQC  
Large Scale Subjective Video Quality Study  
2018年UT Autstin做的，585个视频，80 个不同的用户使用 101 个不同的设备（43 个设备型号）拍摄，分辨率帧率不统一，算是比较真实丰富的。  
AMT平台众包搞了205000 个意见评分，平均每个视频有 240 个人工评价，评分是0-100。  
一个受试者看7个训练视频+43个测试视频（含4个LIVE-VQA？数据集的黄金视频）  
大概5.5G，倒是比较小。  
* Youtube-UGC  
2020年Youtube提供的一个精选UGC数据集，大约1500个视频每个20s，UGC的视频包含各种类型，很强的一点是每个视频都支持了多种分辨率。  
数据来源是YouTube上有common creative标识的150w个视频，这个总数据集相当大了，采样其实是挺难的事情。首先考虑了内容类别，搞了15个类别，然后要求分辨率不同，选了不同分辨率的。然后具体采样做法是考虑了空间、时间、颜色、块变化4个维度的指标，然后做了网格采样？  
每个视频都有100+的主观评分，是1-5的值。还人工打了内容类型标签。  
官网上有下载链接不知道能不能下，YUV的格式2T、H.264 110G、VP9的只有20G。  
* LSVQ  
2020年还是UT Austin做的一个目前数量最大的VQA数据集，真的好有钱。包含39,000 个现实世界的失真视频和 117,000 个时空局部视频补丁（“v-patches”，TODO：这是什么？）。  
有6284人参与实验，共 5.5M 人类感知质量注释。  
原论文Github上有下载工具，听说可能下不全。自己试了下还真是有问题，分了两部分，第一部分是给了colab脚本去下载，速度非常慢……第二部分还是box网盘填写信息密码发邮箱，但是收不到密码……Hugging Face上teo wu看传了一份，大概不到100G可以下。  
数据集构建很有意思，视频是IA数据集和YFCC100M中粗筛了40w的视频，每个截取7s。然后找了19k的社交网站UGC视频，可能涉及版权问题，没有直接用UGC视频，而是计算了26个视频指标，从40w视频中采样了最后的39000个视频，采样的规则是使得26个视频指标的分布和19kUGC视频相似。这样就认为采样除了一个接近UGC视频的数据集。  
也是众包的方式，所以也要做测试拒接不合格的受试者，检查了卡顿、评分相似性等，最重要的是混入了4个重复视频和4个KoNVid数据集的视频作为参考，如果评分误差比较大就会拒绝，拒绝了1046个，按比例看还挺多差不多1/7的。  
* MSU-VQMB  
就36个视频，不过都是比较高清的。  
总共收集了来自 10,800 个参与者的 766,362 个有效评价。  
* DIVIDE-3k  
2022年Weisi Lin老师团队考虑审美分数新建的数据集，一开始是3590个视频，做的经典模型Dover。后面2023年继续拓展做ExplainableVQA，在原来的基础上增加了视频数量搞成3634个训练视频，909个验证视频。（也可以叫做Maxwell数据集，或者DIVIDE-Maxwell数据集）  视频长度不固定平均在10s不超过12s。  
每个视频都有审美分数、技术分数和整体分数。后面又补充了13个具体维度的评价（方便做可解释性），包括技术方面的锐度、对焦、噪声、运动模糊、闪烁、曝光、压缩伪影、流畅度，审美层面的内容、构图、色彩、光照、运镜。  
视频来源是YFCC100M和Kinetic-400一起搞的40w视频，去除了声音，采样时考虑了空间、时间、语义三个维度的指标，看了分布，然后进行随机采样，保证分布和整体分布是类似的。  
主观实验应该是找的固定35位学生，因为涉及多维度比较复杂的评分，经过了完整的培训流程。检查机制也是准备了一些有标签的视频进行测试，如果评分误差超过+-1就不能继续。最后选出来的受试者基本上每个人都评了全部数据集，量还是比较大的，  
TODO  
* YouTube SFV+HDR Quality数据集  
2024年YouTube做的，针对UGC竖屏短视频，并且还尝试分析一部分HDR视频，不过发现HDR视频比较少，种类也不全面  
https://media.withyoutube.com/sfv-hdr  
有2030个SDR视频和2000个HDR视频，截取5s，视频都是比较新的，SDR视频是从近期上传的80000个带creative common标签的SDR视频的大的池子里采样的，采样要考虑的指标太多了，首先考虑了内容，找了10个热门类别，然后考虑3个特征（空间信息、时间信息、感知质量），然后根据均值每个维度划分2个区间，一共8个区间，每个类别每个区间取50个，得到4000个，然后人工筛选了2030个。HDR视频就全用了，不过类别不均匀，而且实验对设备要求也高，所以主观实验10个类别每个只选了30个，其他干脆都是直接HDR转SDR评测了一遍。  
找了一个专门的60+人的数据标注团队，每次实验300个视频，大概标注半个小时。一个人标记7次实验2100个，所以平均每个视频大概25-40个评分。  
  
  
### PCQA数据集  
WPC、SJTU-PQA、M-PCCD、IRPC  
SIAT-PCQD、PointXR，一般点云质量评估是固定观察距离的图片/可旋转视角去评估，这两个是支持6DoF移动观察的  
#### WPC  
Waterloo Point Cloud，滑铁卢大学发的一个比较大的彩色点云质量评估数据集，对应论文Perceptual Quality Assessment of Colored 3D Point Clouds（2021）  
### SJTU-PQA  
https://smt.sjtu.edu.cn/database/point-cloud-subjective-assessment-database/  
10个原始模型，7种失真6个级别  
  
### MeshQA数据集  
首先是几何网格  
* LIRIS/EPFL general-purpose  
4个参考网格，84个失真网格（平滑、加性噪声）  
* LIRIS masking  
4个参考网格，24个失真网格（局部加性噪声）  
* UWB compression  
5个参考网格，63个失真网格（噪声、平滑、压缩）  
* IEETA simplification  
5个参考网格，30个失真网格（简化算法）  
  
然后是纹理+几何网格  
* TMQA  
55个参考网格，3000个失真网格（其实搞了343k个失真网格，评分不过来就挑了下）  
* CMDM-TMQA  
5个参考网格，80个失真网格  
* SJTU-TMQA数据集  
21个参考网格，945个失真网格（简化、量化、加性高斯噪声、纹理子采样、纹理压缩）  
  
## 主观实验方法  
受试者——专家非专家？理论上覆盖更多参与者，获取统计上的均值是更好的方式，ITU -RBT.500-14反正建议15人以上  
筛选——受试者是否合格？打分太接近或者极端，和部分含参考分数的视频评分相差很大  
培训——评分标准是什么？给出最高质量视频参考，学习不同的失真类型，了解各种影响评分/不影响评分的项目（这个其实很重要，例如短期卡顿扣不扣分），有练习阶段熟悉评分范围  
实验条件——是否要所有受试者一致？下载视频避免网络播放卡顿，确认设备条件足够流畅播放，屏幕大小、色彩设置，室内光线，观看距离、角度，播放器页面没有影响主观评价的信息，实验室环境，播放时间间隔，整个实验时长，是否能在同一天完成避免状态变化  
数据清洗——不合格的受试者，不服从正态分布的异常值  
  
  
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
最终的效果就是拿到了图像某一区域的权重信息（可能二值也可能多级），有了这个信息就可以做分层采样、分区域评分增加权重等等操作。还可以结合CAM图做可视化补充可解释性分析  
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
### 回归头  
提取特征后过回归头获得IQA/VQA评分，这里似乎没有太多可以做的，传统的就是支持向量回归、随机森林回归，基本没见使用  
现在基本是MLP回归头，并且不用做很深。两路MLP回归头一路预测分块权重一路预测分块分数最后加权求和也是常见做法  
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
#### OU-BIQA  
无主观评分训练数据做NR IQA  
#### Learn to Rank  
获取绝对评分困难，但是比较两幅图像优劣相对而言简单，有这个信息也可以间接进行评分  
#### 回归问题转分类问题  
Regression via classification  
部分数据集是评分分布而不是MOS均值的形式，可以直接预测评分分布  
### 跨数据集问题  
不同IQA数据集的图像、MOS分布不一样，很难放在一起训练/验证。并且在单一数据集训练过拟合，跨数据集性能一般也很差。如何联合多个数据集，获得较好的泛化性能还是很有意义的  
### 数据增强  
感性的结论是VQA任务不应该做数据增强。经典的一些方法例如加噪声、变颜色、Mixup等方法应该都是不能用的，对画面质量会有影响。  
一些空间变换如翻转、裁剪、resize理论上可能有效果。但负面影响未知。  
视频画质增强、生成模型绘制等方式明显也会改变质量评分。  
那有没有能用的？用了之后效果如何？其实是值得分析的问题。  
#### JND  
最小可察觉误差JND，把来自人眼视觉系统特性&心理效应的视觉冗余量化出来，可以指导编码和质量评估。通过JND来生成一些评分接近的训练数据做增强  
#### 量化相对得分  
分析resize、crop、sharpen、CutMix等一些图像处理操作对图像质量评分的相对影响，在对原数据进行一些操作进行数据增强时，可以根据相对得分计算出一些假的分值，或许可以有用  
这个是不是和zero-shot相关，通过数据的关系去推测标签的关系？  
### 图像增强+IQA  
图像增强模型一般包含了一些质量相关的信息。利用图像增强模型预提取特征；使用图像增强模型为NR-IQA获得伪参考图像……可能有利于IQA任务  
### 小数据集/零样本学习  
#### 大数据预训练+小数据集微调  
大数据集LSVQ上预训练的再放到小数据集上微调  
#### 相似任务预训练+小数据集微调  
用数据量多的视频理解、失真类型&级别分类等任务预训练的模型作为底层模型，提取出公共的特征  
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
可以使用比较新的环境，下面的版本可以跑，就是numpy版本2.0新一点的话有个包里的sctypes用不了，注释掉就好  
```  
conda create --name iqapytorch python=3.12  
conda activate iqapytorch  
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia  
```  
装包时本来还想着最小化安装，一些requirements里面不重要的包就不装了，结果setup.py的时候全自动装上了，所以没啥说的  
```  
pip install opencv-python-headless pandas pillow scipy scikit-learn tensorboard timm tqdm future einops（没必要手动安下）  
git clone https://github.com/chaofengc/IQA-PyTorch.git  
cd IQA-PyTorch  
pip install -r requirements.txt（新版本可能有些自己要手动处理的，例如pip install datasets pre-commit pytest accelerate==1.1.0 ruff）  
python setup.py develop  
```  
2. 准备数据集  
常用IQA数据集都上传到Hugging Face了，真是一件大好事  
https://huggingface.co/datasets/chaofengc/IQA-Toolbox-Datasets  
使用的时候下载数据集对应压缩包，推荐在代码目录创建datasets目录，然后解压到这里，可以匹配配置  
除了数据集本身，所有数据集的标签文件和数据集划分设置，都打包到了一个meta_info.tgz文件，这个也要一起下载了解压到datasets目录  
如果服务器可以访问Hugging Face，可以直接跑python脚本下载数据集，示例脚本项目README.md里有，在IQA-PyTorch目录下执行就行，想下什么数据集就改filename名就行  
```  
import os  
from huggingface_hub import snapshot_download  
  
save_dir = './datasets'  
os.makedirs(save_dir, exist_ok=True)  
  
filename = "meta_info.tgz"  
snapshot_download("chaofengc/IQA-Toolbox-Datasets", repo_type="dataset", local_dir=save_dir, allow_patterns=filename, local_dir_use_symlinks=False)  
  
os.system(f"tar -xzvf {save_dir}/{filename} -C {save_dir}")  
```  
除此之外还可以关注下DataLoader的实现，基类定义在  
pyiqa/data/base_iqa_dataset.py中，派生出通用的NR数据集DataLoader定义在pyiqa/data/general_nr_dataset.py中，然后一些数据集还需要有特殊的处理的要再自己派生实现DataLoader类  
关于数据集划分比较规范，统一做了seed=123的划分文件，存了train、val、test数据集的数据index，然后DataLoader里会读取划分结果去加载训练集验证集测试集，并且都分了10种split（大部分是10种随机的8:2开划分，比不重叠的5-fold还严谨），可以交叉验证，自己读一下  
```  
f = open("./datasets/meta_info/koniq10k_seed123.pkl",'rb')  
split_dict = pickle.load(f)  
print(split_dict[2]['train'])  
```  
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
  
## VQA  
### Q-Align推理  
```  
conda create --name qalign python=3.10  
conda activate qalign  
  
git clone https://github.com/Q-Future/Q-Align.git  
cd Q-Align  
pip install -e .  
  
# 命令行推理指定文件（会自动从hugging face下载q-future/one-align模型17g左右，也可以自己下载然后--model-path写本地目录）  
python q_align/evaluate/scorer.py --img_path 视频文件路径 --video --model-path q-future/one-align  
```  
### FastVQA推理  
```  
conda create --name fastvqa python=3.10  
conda activate fastvqa  
git clone htps://github.com/QualityAssessment/FAST-VQA-and-FasterVQA.git  
cd FAST-VQA-and-FasterVQA  
pip install -e .  
```  
下载预训练模型放到./pretrained_weights/目录下（注意有个文件名问题，1\*1的\*可能被改为下划线，linux下要用转义\\\*），然后执行推理脚本，可以设置用那个模型，FastVQA最好最慢，FasterVQA适中，还有一些更简化的版本。用CPU也能跑挺快的。这里有个bug，有的机器上会出现，就是因为这个脚本是先导入自己的model再导入torch，可能就会无法load，需要交换一下顺序  
```  
import torch  
from fastvqa.models import DiViDeAddEvaluator  
```  
```  
python vqa.py -m FasterVQA-MT -v [YOUR_INPUT_FILE_PATH]  
```  
由于默认进行随机网格采样，所以每次执行结果会波动，如果要固定采样，可以修改fastvqa/datasets/fusion_datasets.py文件中get_spatial_fragments的处理  
```  
    elif fix_sample:  
        if hlength > fsize_h:  
            rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned), dtype=torch.int)  
        else:  
            rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned), dtype=torch.int)  
        if wlength > fsize_w:  
            rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned), dtype=torch.int)  
        else:  
            rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned), dtype=torch.int)  
```  
  
# 论文整理  
## IQA  
看了很多经典IQA模型有一点总结。  
核心的问题：  
一方面是分类任务预训练模型提取特征在IQA任务上微调效果不够好，说明这些“通用视觉特征”难以完全区分质量差异，需要寻找更好的“质量特征”。（即使ImageNet预训练的ResNet输出加回归头在IQA数据集上微调效果也一般，FR-IQA比NR-IQA更差）  
而另一方面由于IQA数据集大小有限，直接在IQA数据集上训练模型虽然说能直接获得一些“质量特征”，但也效果不够好。（有的FR-IQA还可以，但NR-IQA都很一般）  
解决思路：  
一种方式是“精炼分层特征”，实践证明CNN提取出的分层特征/ViT提取出的特征中，是有一部分可以用于区分质量差异的，这些特征再经过一些网络，在IQA数据集上微调，就可以选出有用的。  
另一种方法是“融合质量特征”，通过质量相关任务（如失真类别&级别分类，质量评分比较，图像增强）预训练的模型会获得一些和质量相关的特征，但缺失“通用视觉特征”，这时只要将二者特征融合，就能够取得很好的效果。DBCNN是最有代表性、最简洁也效果非常好的，后续很多网络都是类似的思路只不过网络做的更复杂。  
  
### （13.12.16同济 经典FR客观指标）VSI: A Visual Saliency-Induced Index for Perceptual Image Quality Assessment  
经典客观FR IQA方法，计算视觉显著性map（边缘、色彩突出的区域），然后算参考图和失真图的差距，性能就还不错（主要是TID 2013有0.9的PLCC，其他数据集倒是没有明显优势）。原理上算是SSIM的加权拓展，认为不同区域对质量影响的权重不同，要按照视觉显著性加权。  
  
  
### （17.1.13德国Fraunhofer研究所 DeepIQA/WaDIQaM）Deep Neural Networks for No-Reference and Full-Reference Image Quality Assessment  
非常经典的早期DL IQA模型，结构是很工整的，分开做了FR和NR的，单独看NR的吧  
特征提取部分就是（2层CNN+1层2x2最大池化）x5，图像拆分为32x32的小补丁输入，所以经过卷积之后正好size为1，通道升到512维。  
回归Head是分了预测和权重两路，每个小补丁预测分数，并预测补丁权重，最后加权平均获得整个图像的分数。  
跑分在FR数据集上还可以，但是在NR数据集上比较差了。  
不限制输入大小，训练的时候是每张图像随机crop 32个32x32的小补丁，测试的时候是整张图像不重叠的切补丁输入。  
  
  
### （17.7.26巴塞罗那自治大学）RankIQA: Learning from Rankings for No-reference Image Quality Assessment  
使用Learn to Rank预训练，生成一些退化后的图像可以知道和原图像比较的排序，这样就可以以图像对的形式输入孪生网络Siamese Network进行训练，认为可以拿到一些区分图像质量的特征，训练好的网络作为backbone再加回归头微调  
比较早期的工作，指标应该挺一般的，后续也没什么后续的工作，具体效果如何有待进一部分分析  
TODO  
  
### （17.9.15谷歌）NIMA: Neural Image Assessment  
早期DL IQA模型，网络结构上没有任何特别的，就是直接分类任务预训练CNN（Inception-v2和VGG16）换个分类/回归头。只跑了AVA、LIVE、TID2013数据集，指标也挺差。  
特别之处在于对AVA数据集的处理，因为这个数据集给的其实是1-10几个评分级别的投票分布，所以模型是接了个分类头去预测10个类别的概率。这种“有序分类任务”，直接用交叉熵丢失信息，当回归任务处理也效果不好，可以使用特殊的损失函数，这里用了EMD？损失函数。这种分类模型做回归的做法值得学习。  
  
### （18.1.11伯克利 LPIPS）The Unreasonable Effectiveness of Deep Features as a Perceptual Metric  
经典方法，思路非常简单，就是用固定的神经网络（例如ImageNet预训练VGG）提取原图和失真图特征向量后比较相似性，是一种FR方法，并且相对而言是一种客观指标，无需进行模型训练。  
做出来发现效果非常好，用深层特征比相似性更符合人类感知，比SSIM、PSNR这样的“浅层”特征相似性指标，更接近HVS中两张图像的相似性，可以用于IQA任务。  
  
### （19.2.17巴塞罗那自治大学 MT-RankIQA）Exploiting Unlabeled Data in CNNs by Self-supervised Learning to Rank  
19年RankIQA的改进，加了多任务  
TODO  
  
### （19.6.18 西安电子科大）Personality-Assisted Multi-Task Learning for Generic and Personalized Image Aesthetics Assessment  
个性化图像美学评价，考虑了受试者的BF 大5人格特质。  
模型一阶段进行预训练，孪生网络同主干双Head做多任务交替训练。任务一是通常的输入一张图像预测美学评分，任务二比较特殊，首先需要收集不同个性受试者喜爱的图像，然后让模型输入图像预测喜爱的受试者个性。认为这样获得的一个网络就有了通用的预测美学评分&受试者个性的能力（输入1张图片，给出1个分数+5个个性分数共6项输出），还可以把美学评分和个性关联起来。  
模型二阶段做个性化微调，加个融合层，最后评分是由6个分数加权得到的，对具体的用户收集一个小的微调训练集，然后微调融合层的权重。  
做法有点意思，但是我其实不太理解Task2的设计，预测出的受试者个性似乎只是一个统计均值，然后最终分数是直接每个维度个性分数来加权，那前提假设就是喜欢一张图片的人个性在5个维度上独立且集中分布，5个维度和审美评分线性相关，我认为这两个假设都存疑，所以这个做法的合理性直观感觉不太够。  
  
### （19.7.5武大DBCNN）Blind Image Quality Assessment Using A Deep Bilinear CNN  
效果很不错的CNN模型  
网络结构分了两路：  
* S-CNN 合成失真信息  
一路是只保留Backbone的S-CNN合成失真分类任务预训练模型（S-CNN模型就是稍微调了下的VGG-16，预训练数据集是4744张图像的Waterloo Exploration数据集和17125张图像的PASCAL VOC数据集自己做了9种合成失真后得到的）  
* VGG-16 真实失真信息（其实就是通用视觉特征）  
一路是只保留Backbone的VGG-16 ImageNet分类预训练模型  
  
然后特征融合，结构上相当于合并了两个预训练CV模型。  
两路的合并方式是bilinear pooling，是一种挺不错的经典特征融合方法，特征直接拼接或者加权相加实现的是一阶交互，双线性池化是二阶的，两个向量外积再池化，得到512x128的特征图。  
回归Head就是一层线性回归层，512x128->1  
S-CNN和VGG-16本身应该都是限制224x224的，但是合在一起做了池化应该是不限制输入大小的，所以实际训练的时候就是随机crop到448/384，验证好像就是原图  
  
### （19.10.15pub NTU）SGDNet: An end-to-end saliency-guided deep neural network for no-reference image quality assessment  
特征图出来乘了显著性map，认为人会更关注显著性区域的失真，会有一点增益  
但是后续很多工作并没有做ROI，个人认为ROI在IQA中的作用不太直接，因为质量感知的ROI可能和视觉显著性的ROI不完全一致，显式引入不一定通用  
  
### （19.12.20德州大学）From Patches to Pictures (PaQ-2-PiQ): Mapping the Perceptual Space of Picture Quality  
做了一个较大的in-the-wild IQA数据集（应该是FLIVE），并且每张图像还带了3个不同大小patch（20% 30% 40%）的MOS，从而可以分析下局部和整体的质量关系，答案是比较相关（算是随机crop的理论基础了）  
做了个模型就是ResNet-18改改，因为有了patch评分的信息，所以引入了ROIPool加权，把patch质量评估结果和全局质量评估结果一起考虑得到最后得分，会比光输入图片高一些  
感觉这也不太算是数据增强的技术，patch也是做了主观评分实验的……  
  
### （20.1.20康斯坦茨大学）DeepFL-IQA: Weak Supervision for Deep IQA Feature Learning  
这篇文章主要是做了KADID-10k数据集所以很重要。  
但其实这个论文也提了个DeepFL-IQA方法，也还挺有意思，思路是使用大量有FR-IQA评分的数据来弱监督训练NR-IQA，所以同时还搞了个KADIS-700k数据集。这个数据集是140000张原始图像，然后随机每个找一种失真降级5个级别，所以有700k个失真图像，但没有主观评分，有11种FR-IQA方法的客观评分。  
TODO：具体的方法还是不太确定细节  
第一阶段预训练，是在KADIS-700k数据集上做多任务训练，同时预测11个FR-IQA的结果，认为这个预训练模型就可以提取到很有用的特征。第二阶段用预训练模型结合多尺度空间池化MSLP提取图像特征，加一个回归头预测分数。  
也就是说，用足够多质量评估相关的任务预训练出来的模型就能很好的获取质量相关特征。  
这个工作在KADID-10k上直接跑出0.936的高分了，这种评分得MANIQA这种才能打，其他数据集上没有这么夸张也还是比较高的，在当时绝对是SOTA了，后续工作很少对比这个，不知道是不是这个原因。  
  
### （20.3.7西工大 HyperIQA）Blindly Assess Image Quality in the Wild Guided by a Self-Adaptive Hyper Network  
TODO：思路没太看明白，大致是把ResNet的高层语义特征和底层纹理特征分开处理了  
ResNet之后高层语义特征分了一路输入啥内容理解网络，分层特征输入MLP。然后内容理解网络的分层输出又输入MLP，反正最后出的结果抽象理解下就是综合了高层和底层的特征，把CNN用的比较充分了  
输入是一张图像随机crop 25张224x224然后结果取平均  
  
### （20.4.11西安电子科大）MetaIQA: Deep Meta-learning for No-Reference Image Quality Assessment  
好像是在合成数据集上训练然后去跑真实失真数据集，普通CNN模型  
合成失真数据集上还可以，但是真实失真数据集上指标挺差，感觉参考价值有限  
  
### （20.4.16港城大 DISTS）Image Quality Assessment Unifying Structure and Texture Similarity  
对视觉纹理重采样不敏感的全参考 IQA 方法，设计思想也是基于HVS感知时对于细节纹理的变化没有那么敏感（举了例子是草地，压缩会导致像素上变化挺大，但是一顿乱的细节纹理只要大致结构差不多，人看不出很大差距）  
也是用的LPIPS这样的深度特征差异来比较相似性，但是做的更进一步，提升了可解释性很好。思想是把深度特征的均值作为结构特征，深度特征的方差作为纹理特征（做了很多推导有些道理，高层深度特征本来就是和一些语义形状有关）。用的是VGG的分层特征。  
论文做的很好补充内容扎实，发的TPAMI，值得学习  
  
### （20.5.18西安电子科大 AIGQA）Blind Image Quality Assessment With Active Inference  
图像增强+IQA  
使用GAN网络获得增强后的图像，然后和原图作差得到失真图，和原图求相似性得到结构退化图。四个图作为输入过CNN+MLP得回归结果  
说法是用GAN网络获得增强图像来模拟人类视觉看到图像后有个主动推理的过程，叫啥IGM内部生成机制，可能就是会想象出应该有的样子做FR？  
比较早的工作跑分挺一般的，直观感觉这种很直接的处理实际还是没引入多少有用的特征，也取决于GAN网络的增强效果，应该没有很好  
  
### （20.6.10天津大学）VCRNet: Visual Compensation Restoration Network for No-Reference Image Quality Assessment  
图像增强+IQA，之前有一些通过GAN进行图像增强获得“参考图像”然后分析原图和参考图之间差异来做IQA的工作，但问题是GAN生成的参考图像不靠谱，其实有问题，拿来做IQA效果也不稳定。  
所以这里没有用基于GAN的图像增强网络，而是做了一个改了下的U-Net图像增强（恢复）网络，从中取分层特征也会比较方便。最后用图像增强网络Decoder输出的分层特征融合EfficientNet的分层特征做IQA，效果还不错。  
  
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
  
### （21.5.13中科大）GraphIQA: Learning Distortion Graph Representations for Blind Image Quality Assessment  
比较特别的做法，用的图表示学习。这方面不是很懂还是要另外学习下。  
大致的思路首先把原始图像转为失真图表示DGR，然后再用图网络去做回归任务。  
看指标还是不错的，和中期的MetaIQA、HyperIQA都差不多。  
  
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
  
### （22.2.24NTU）Distilling Knowledge From Object Classification to Aesthetics Assessment  
搞的美学评估，跑的AVA数据集。  
做的东西巨简单，但是故事讲的还挺有道理，写法值得借鉴，一开始一堆名词把我看懵逼了。  
背景是认为直接拿分类任务预训练模型提取的特征（定义为GSF通用语义特征）做美学评分任务效果不好，没有直接在美学评分数据集上训练获得的美学特征效果好，说明通用语义特征不能很好地用于美学评估。  
但是认为通用语义特征中包含用于美学评估的信息，所以搞了多个分类任务预训练模型，输出特征做多层空间池化（MLSP）连起来。然后用BN层+三层MLP预测结果就好了（多个预训练模型特征拼个回归头）。那认为第二层MLP输出的特征就是从通用语义特征中提取出来的美学特征。  
再做了个知识蒸馏，简单网络学第二层MLP输出的美学特征和第三层MLP输出的预测结果，效果也很不错。  
  
### （22.3.31Oppo 个性化美学评估PARA数据集）Personalized Image Aesthetics Assessment with Rich Attributes  
31,220 张图像， 438 名受试者  
很特别的点在于除了图像的客观属性标签 构图、光线、颜色、景深、对象强调、内容 等  
还采集了主观属性标签：1） 内容偏好，2） 判断难度，3） 情感，4） 分享意愿  
特别是还有每个受试者的个性化标签：（a）年龄，（b）性别，（c）教育经历，（d）艺术经验，（e）摄影经验，（f）大五人格特征（一种心理测试来的，类似MBTI）  
  
这样就有可能做很多分析了，打这么丰富的标签有点厉害  
  
### （22.4.29清华）MANIQA Multi-dimension Attention Network for No-Reference Image Quality Assessment  
非常经典的Transformer IQA模型，指标爆杀之前所有模型，把注意力机制用的很好，值得学习  
网络结构是直接ViT预训练模型，取了第7、8、9、10一共4层的分层特征（总共12层Block）。分层特征先过Transposed Attention Block（TAB）实现不同通道？之间的融合，然后reshape成特征图过了卷积和Swin Transformer组成的Block，实现空间的融合？最后过一个两路的双层MLP，分别获得patch的分数和权重，最后加权求和得到整个图像的评分  
输入是训练阶段一张图像随机crop一张224x224（反正多轮训练），测试阶段一张图像随机crop 20张224x224然后结果取平均  
  
### （22.7.25NTU CLIP-IQA）Exploring CLIP for Assessing the Look and Feel of Images  
早期MLLM做IQA任务的工作，思路简单直接，计算图像Embedding和文本Embedding的余弦相似度来得到质量评分。  
文本尝试了一些不同的方式，例如“This is a high/low quality photo”,“This is a photo of high/low quality”等，最后发现最简单的“good/bad photo”是最好的，而且good/bad正反去比较比单独用good比较效果更好，思路有点意思。  
  
### （22.7.29港城大）Image Quality Assessment: Integrating Model-Centric and Data-Centric Approaches  
不是Model的文章，而是分析问题的文章。实验评估了一些IQA模型的过拟合问题，以及一些数据集过于简单无法获取多种失真特征的问题。能分析这个问题还是挺有意义的。  
认为既然发现有一些图像对于提升模型的泛化性没啥意义，那就应该增加多样性约束，选出更复杂、更多样性的样本，这样才能训练出好的模型。提出了采样更具多样性的图像样本的方法，没有细看，具体可以再分析下。  
  
### （22.8.25上交 StairIQA）Blind Quality Assessment for in-the-Wild Images via Hierarchical Feature Fusion and Iterative Mixed Database Training  
两个创新点都有点意思  
一是把分层特征做了更复杂的一个融合方法，不是分开处理，而是逐层的合并。处理流程像是阶梯状，第一层分层特征过个卷积，加上第二层分层特征后再过个卷积，加上第三次分层特征后再过个卷积，加上第四次分层特征后再过个卷积；同理2、3、4的加；3、4的加；4的单独加。最后能拿到4路融合后的分层特征，再加回到正常第5层输出的高级特征。看消融实验结果还是有点用，不是很大。这里我理解应该没有分层特征分出来再过注意力block这种做法性能好，但是结构上是简单规则的，可以直接理解，最后还是只输出一份特征，是一个融合和低层、高层特征的结果，比直接用分类模型的高层特征做IQA还是要好一些，相当于增加了低层特征的权重吧。  
二是多个数据集的数据放在一起训练，这个看起来做的还不错，多数据集训练本身也是不太好做的一个点。这里做法比较简单，就是特征提取的backbone是多个数据集训练的，回归头是每个数据集上训练的。具体迭代流程要看代码。  
  
### （22.10.10pub上交）Image Quality Assessment: From Mean Opinion Score to Opinion Score Distribution  
比较特别的工作，内容和标题一样是将预测MOS（平均评分）改为预测OSD（评分分布）。之前比较经典的就是NIMA在审美评分AVA数据集上做的。但是由于通用IQA数据集这边很少有OSD信息，所以不好做，只有KonIQ有。作者还自己重建了LIVE数据集加了OSD信息，就有两个数据集可以跑了。  
模型结构应该也是和NIMA的最简单直接的做法很像，NIMA就是直接用经典CV分类模型（ResNet、VGG、Inception等）提取的特征，接MLP做N个评分级别的分类，然后用过了SoftMax之后的结果作为预测概率分布对应实际的概率分布。这里做的一个改进是提取的特征又过了一个特征模糊层再接MLP。这个应该是控制里面常用的模糊理论啥的，我不太了解，具体实现看论文结构图和源代码应该就是一个SoftHistogram软直方图层，做了1D卷积+绝对值+1D卷积+ReLU，感觉和接了卷积层有点像。  
直观感觉这种只用了CNN高层特征的模型应该结果不会好，不过跑OSD指标本身就没啥baseline对比，主要是自己对比了下不同的CNN backbone，那加了卷积层肯定结果更好。也跑了MOS的SROCC比DBCNN优？感觉存疑。  
  
  
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
  
### （23.5.5南京邮电大学 DPNet）Learning Hybrid Representations of Semantics and Distortion for Blind Image Quality Assessment  
没查到接收日期，应该22年的  
想法挺好，是用多任务训练的方法想办法用单个网络同时学到语义表示和失真表示，不需要像大部分其他IQA模型一样融合两个网络特征  
做法用知识蒸馏学ResNet152的特征，从而学语义表示；然后接MLP跑失真分类任务微调学失真表示。最后再接另一个MLP去预测IQA分数。  
TODO：没仔细研究这个训练过程，如果是分阶段还叫多任务吗？学失真表示的时候会不会影响语义表示的记忆了？  
看指标整体一般，但是在KADID-10k上SROCC 0.923是不错的成绩了，直观感觉应该跑不过两个网络结合的模型才对，但是没开源代码分析不了……  
  
### （23.6.10港城大 FLRE）Towards Thousands to One Reference: Can We Trust the Reference Image for Quality Assessment?  
FR-IQA的工作，认为参考图像不唯一，可以在参考图像的等质量空间中选取最佳的参考图像，研究了生成等质量空间和搜索最佳伪参考图像的算法。  
  
### （23.8.6NTU）TOPIQ A Top-down Approach from Semantics to  Distortions for Image Quality Assessment  
TODO：模型稍有点复杂没太研究明白  
核心的一个思路是考虑高层特征中包含的语义信息，然后用高层特征指导自顶向下指导底层特征，特征连线有点复杂，主要的改动都是在一些特征融合模块上，引入了gated local pooling block（GLP），self-attention block（SA），cross-scale attention block（CSA）三种模块  
分了FR和NR两种模式，FR就要多融合一些参考图像的信息，看跑分还是相当高的  
回归Head是一个三层的MLP，感觉不算很重要了，前面还有注意力模块，整体已经很复杂了  
输入应该是任意size的，毕竟有GLP池化统一特征图大小，不过训练的时候针对不同数据集还是做了随机crop到384或者224，原图很大也会resize到448/384-416  
  
### （23.9.24广州大学）CDINet Content Distortion Interaction Network for Blind Image Quality Assessment  
图像增强+IQA。主要参考了VCRNet的工作，区别是VCRNet中直接用的U-Net图像增强网络Decoder部分分层特征，这里是又做了一路Decoder恢复原始图像也得到分层特征，那增强和原始的分层特征作差其实得到的就是差分的“失真特征”。  
然后和CNN直接提取的特征做融合的时候也不是直接加，而是用了注意力模块。  
跑分看起来还可以，不过也和MANIQA有差距。代码没开源。  
  
### （23.10.9山东师大）MFCQA Multi-Range Feature Cross-Attention Mechanism for no-reference image quality assessment  
主要是做模型的设计，特别之处是融合了3个不同range的特征。  
这个multi-range的概念查了下还是attention-based模型中经常提到的术语，我理解其实和multi-scale在本质上是相通的。multi-scale形容的对象主要是客观的图片分辨率/特征尺度，空间大小上不同。multi-range形容的对象主要是主观的注意力/感受野/上下文依赖，覆盖范围大或者小。说是受到人类视觉机制的启发，人眼视网膜会既关注特定区域，也关注周边区域。（讲的有点抽象，我理解就是全局信息+局部信息）  
具体做的时候用了三个特征提取器：  
（1）预训练ViT 7、8、9、10层输出特征（同MANIQA），认为是提取的全局特征  
（2）Pale Transformer，认为是提取的次全局特征  
（3）VGG-16移除Head再接个CNN调整输出尺度，认为是提取的局部特征  
然后三通道的特征过交叉注意力的模块，最后过分数+权重的双通道回归头，算是比较常见的处理了。  
指标很高，超了MANIQA，在MANIQA基础上另外加了两路特征应该合理，不过似乎没有开源代码。  
操作上其实就是用了3个模型的特征，讲故事上说全局、次全局、局部这里个人认为没有什么道理，将CNN认为是局部、ViT认为是全局不合适。  
  
### （23.10.20佛罗伦萨大学）ARNIQA Learning Distortion Manifold for Image Quality Assessment  
  
### （23.10.21重庆邮电大学 PINet）Blind Image Quality Assessment With Coarse-Grained Perception Construction and Fine-Grained Interaction Learning  
注意力特征融合做的比较特别。  
网络结构是先用了StairIQA的Backbone，获取了一个分层特征融合比较好的特征图。然后分两路ResNet+MLP，预训练阶段用合成失真数据集训练，一路用两张图像输入比较质量好坏的任务训练获取质量相关特征，一路用失真类型分类任务训练获取失真相关特征。  
微调阶段用真实失真数据集训练，是把上面两路ResNet的分层特征用注意力机制连接起来（这里的连接方式还挺有意思，相当于每一层混合后输入下一层），说是这样模拟人脑同时处理质量和失真信息。  
最后跑分不算高还可以，专门做了FG-IQA的验证，不过只是分了高质量和低质量两半（分太多肯定分不好看），看起来还可以，比DBCNN、HyperIQA好一点  
  
  
### （23.12.12南京大学SaTQA）Transformer-based No-Reference Image Quality Assessment via Supervised Contrastive Learning  
网络结构还是比较复杂的，做的比较长。整体分两路，一路提取失真Distortion特征R，一路提取退化特征P。  
首先是失真特征提取，做了一个融合CNN和Transfomer的Multi-Stream Block。图像先过ViT得到特征图F，然后又分三路：  
第一路过DeformConv和线性层（CNN Low-Level）  
第二路过depthwise卷积层和最大池化（CNN Low-Level）  
第三路过多头自注意力和线性层（Transfomer Low-Level）  
最后三路结果合起来得到的F'再和ViT直接得到的特征图F过CBAM注意力模块的结果做差分。这样的MSB块还搞了三个，前两个还有下采样。最终得到失真特征R。  
另一路退化特征提取，网络就是ResNet50+MLP，但是做了对比学习：很类似CONTRIQUE的做法，预训练阶段做自监督学习，方法是对比学习，具体任务也是失真图像的失真类型&失真程度分类，和CONTRIQUE的区别是只用合成数据集中的参考图像不用UGC图像（这也算进步吗？）。最后只使用Backbone部分预提取出退化特征P。  
还用了个Patch注意力块进行特征融合，把两路提取的特征P和R拼在一起，再去做回归。应该算是模型魔改玩的相当6的网络了。  
跑分结果也是相当好，全面超MANIQA的。  
预处理上应该训练和测试都是一张图像输入8个random crop的224x224块  
  
### （23.12.14pub港中文 DepictQA）Depicting Beyond Scores: Advancing Image Quality Assessment Through Multi-modal Language Models  
用MLLM做可解释的IQA的一个尝试工作，我认为难点在于IQA任务要想解释太复杂了，很难获得足够且格式合适的训练样本。为了简化任务，没有做直接定量评分，设计的工作方式是输入参考图像和2个失真图像A、B，然后问模型A和B哪个质量好，为什么。（FR的比较比NR的单刺激要明确很多）  
训练的时候设计了三类辅助任务，做了对应的训练数据集。一是输入参考图像和失真图像，描述失真图像的质量（有什么失真）；二是比较质量，输入参考图像和2个失真图像，比较失真图像好坏；三是解释比较结果，已知两个失真图像质量比较结果的情况下，再解释为什么是这样。  
数据标注的细节值得学习，描述性语言本身是很多样化不结构化的，但是数据标注的时候人为设计了模板，限制了一些失真类型和级别，再使用GPT4来转化自然语言，成功做出了训练数据集，用于LoRA微调。  
  
### （24.3.16清华 MLLM IQA综述）A comprehensive study of multimodal large language models for image quality assessment  
实验做得还不错的一个MLLM测评工作。设置了9种提示词模式（3种测试方法：单刺激/双刺激/多刺激 x 3种提示词设计：标准、上下文、思维链），在多个IQA数据集上测试了4个经典的MLLM模型：  
LLaVA-v1.6 (Mistral-7B) 、InternLM-XComposer2-VL (InernLM2-7B) 、mPLUG-Owl2 (LLaMA2-7B)、闭源MLLM GPT-4V  
但是结论还是比较浅的，没有特别明确的发现。首先提示词设计对于结果影响很大，然后是GPT-4V效果比较好，很多场景下都和经典专家模型不相上下，开源的3个模型效果很一般。不过开源模型微调后的Q-Instruct效果就很好，还是要微调。算是证明了MLLM的潜力，但是具体使用方法上有很多值得优化的点。  
  
### （24.5.29港中文 DepictQA-wild）Descriptive Image Quality Assessment in the Wild  
DepictQA的扩展工作，形式上一个比较大的改进是既可以输入单图像评估，也可以输入两个图像比较，之前DepictQA只能输入2/3个图像做FR质量评估。  
具体做法是分了4类任务，首先单图输入和双图输入是两个大类，然后各自有简要任务和详细任务：  
单图简要任务 —— 失真识别  
单图详细任务 —— 描述失真的影响  
双图简要任务 —— 比较相对质量  
双图详细任务 —— 描述比较结果的原因  
自建了比较大的合成失真训练集，35个失真类别 x 5个失真级别，并且支持2种同时存在。  
模型训练也有一些细节的优化，但看不太懂。8卡A6000 22个小时。  
最后性能是暴打Q-Instruct，也炒了GPT-4v的  
  
### （24.5.29港城大 Compare2Score）Adaptive Image Quality Assessment via Teaching Large Multimodal Model to Compare  
引入LLM进行比较评分（对比两张图片给出差、较差、相似、较优、优的相对评级），也算是Q-Align模型引入LLM做主观评分的进一步拓展，可以绕开不同数据集之间MOS评分不一致没法混用的问题  
TODO：分析下这里对比评分的用法  
  
### （24.5.29港城大 MDFS）Opinion-Unaware Blind Image Quality Assessment using Multi-Scale Deep Feature Statistics  
这里讨论的是比较小众的OU-BIQA（BIQA就是NR IQA）任务，也就是没有带主观评分标签的训练数据，依然要做IQA，听起来就是没什么依据很难做。  
做法挺有意思的，偏统计的方法。先是用预训练的CNN模型提取分层的图像特征（所谓的MDFS多尺度深度特征提取是金字塔形式下采样，前一层特征做下采样和后一层一致，最后统一特征图大小连接起来），通过统计方法把特征图转换为一个多维高斯分布，也就是实现了图像->高斯分布的映射。  
评分的时候一方面搞了个高质量图像的数据集作为高分的基准，算出来这个数据集平均的高斯分布，然后测试图像也搞成高斯分布，计算两个高斯分布之间的距离得到评分。  
想法很有意思，其实算是造出了FR IQA了。NR IQA中没有对应的参考图，随便找一些高质量图像做参考图？直接比较测试图和参考图的相似性并不好做，毕竟内容都不一样，但是过DNN获得的特征图再转统计学高斯分布表示，再计算相似性就有那么点道理了，可以认为减少了具体的内容信息，而是有一些内容无关的统计量表示，就能通过相似性发现图像退化程度  
  
### （24.6.3清华）UniQA Unified Vision-Language Pre-training for Image Quality and Aesthetic Assessment  
TODO  
拿MLLM做图像质量&审美评分的，好像跑分还挺高。  
  
### （24.6.24Sony 高清IQA数据集）UHD-IQA Benchmark Database: Pushing the Boundaries of Blind Photo Quality Assessment.pdf  
数据集工作，现有的IQA数据集质量分布比较广，大部分还是质量、分辨率比较低，这个数据集是6073张4k分辨率的图片。评分是10个专家评分员每人每个图像评两次得到20个评分，另外数据集还有很多额外的标签，包括标签通过模型和众包标注做的。  
数据池子是Pixabay网站的15w张图片中的收藏数排名前10000张。然后人工剔除了合成图像和过度编辑的图像，剩下6073张。  
严格挑选了10个专家评分员，freelancer众包平台上评分很高，而且有视觉相关背景，基于已有评分的KonX数据集组织了竞赛，选出来的评分员都是准确度很高的。  
10个评分有点少，所以每人评两遍，中间隔了比较长时间避免记住。  
有一段评分标准描述：  
参与者被要求根据可见缺陷（例如噪点、模糊、压缩伪影和色彩失真）造成的干扰程度来评估图像的技术质量。他们被告知，技术质量不同于美学吸引力或吸引力，高技术质量的图像并不总是赏心悦目。然而，参与者也被告知，在某些情况下，例如微距或特写摄影，某些缺陷（例如背景模糊（“散景”））可能是构图的固有组成部分，只有当它们对观察者造成干扰时，才应被视为质量下降。  
  
### （24.8.1上交）No-Reference Image Quality Assessment Obtain MOS From Image Quality Score Distribution  
挺神奇的模型做法，三个优化点吧  
主模型是ResNet+Channel注意力，但是做了3种不同的池化合起来得到特征向量，消融实验看还真是涨点了，是有点意思的涨点方法  
加了个GCN图卷积的分支，直观感觉有点奇怪。是把图像评价的5个level标签作为节点，分析两个标签同时描述一张图片的次数，统计出标签之间的5x5相关性矩阵输入GCN，可以作为一个“映射器”？似乎可以调整主分支的特征权重  
预测的时候是MOS和分布联合预测，也是能涨点的  
最终确实SOTA了  
  
### （24.8pub NTU CMKernel）Continual Learning of Blind Image Quality Assessment with Channel Modulation Kernel  
TODO： 用了个通道调制核，没看明白是啥  
  
### (24.9.9上交 RichIQA）Exploring Rich Subjective Quality Information for Image Quality Assessment in the Wild  
两个主要的创新点。  
一是网络结构拼的比较复杂，分类三个stage，第一段是CNN+Transformer混合网络提取的分层特征，这个是21年CvT: Introducing convolutions to vision transformers中做的。第二段是用了个“长短期记忆网络”（不是LSTM），说的是模拟人脑记忆特性设计的一个神经网络，是22年M-GCN: Brain-inspired memory graph convolutional network for multi-label image recognition这篇文章做的。第三段是预测质量分布DOS，直接第一段分层特征过MLP预测的“算法DOS”和经过第二段得到的“记忆DOS”合在一起作为最终DOS。  
二是不是只预测单个MOS标签，而是预测MOS+SOS（标准差）+DOS（分布）的联合标签，信息更多一些。  
  
### （24.9.23北大 投稿ICLR被拒）Understanding the Generalization of Blind Image Quality Assessment A Theoretical Perspective on Multi-level Quality Features  
理论分析分层特征对IQA模型意义的论文，推导了三个定理，有点点不好理解：  
（1）模型泛化能力边界的表达式  
（2）引入训练集到测试集的分布差异进一步细化（1）的表达式  
（3）模型深度增加，表示能力增加？  
然后解释了定理（1）说明了低层特征对于IQA泛化性的重要性，定理（2）得到的边界比定理（1）更严格，定理（3）说明了高层特征对于IQA泛化性的重要性  
挺理论的，推导看起来还不错，但结论太过于平淡了，是基本的DL领域的共识。审稿意见非常专业地找到了三个定理的原始出处，都是之前CNN理论论文中推导过的  
 Sun S, Chen W, Wang L, et al. On the depth of deep neural networks: A theoretical view. Proceedings of the AAAI Conference on Artificial Intelligence. 2016; 30(1).   
这里放到IQA场景下又推了一遍，说对IQA的工作有指导意义，其实没有新的东西。  
  
### （25.1.15pub河南大学 TD-HRNet）Texture dominated no-reference quality assessment for high resolution image by multi-scale mechanism  
做高分辨率IQA的工作。网络设计挺有思路的，高分辨率图像的处理肯定是整体语义+局部细节  
输入预处理就分了两个尺度：随机不重叠裁剪5个块，并且计算熵给熵最大的块最大权重（纹理多有代表性），还会resize一个低分辨率版本做类似的随机裁剪，实现多尺度。  
然后模型是双分支又区分了整体和局部：局部纹理分支是resnet-18，分层特征做statistic池化，然后拼接成一个整体；整体语义分支是resnet-50，上面3层分层特征做ga门控池化，然后又过比较复杂的注意力模块融合成一个整体（应该是类似TOPIQ的方案）。然后双分支特征再拼接到一起作为最后的特征，过MLP预测质量分。  
训练的时候还用了多任务，是做质量分数预测和质量排序（排序怎么做的？），损失函数就是两个任务的损失加权。  
  
### （25.1.16港中文）Teaching Large Language Models to Regress Accurate Image Quality Scores using Score Distribution  
MLLM IQA输出连续分数的又一个新方法，Q-Align是输出5个离散级别，根据各自预测概率求MOS，本质预测的还是one-hot一个标签，会引入一定误差。这里优化为使用软标签，把一维的质量分数先假设为是连续的高斯分布，然后再转换为离散5个级别的离散高斯分布，然后模型预测5个级别token分布，KL散度算和Ground Truth的离散分布差异？（具体实现可以研究下，不太理解），结果更加准确。  
训练的时候还有保真度的一个处理，实现跨数据集训练，做跨数据集训练的时候不能简单汇总，毕竟分数分布不统一，但可以考虑同一数据集内的排序  
  
### （25.8.18南开）ViDA-UGC: Detailed Image Quality Analysis via Visual Distortion Assessment for UGC Images  
基于MLLM可解释IQA的比较新的数据集工作，建的数据集包含11,534 张图像、36K 畸变边界框和 534K 指令调优数据，分为3个子集分别做三个任务：失真定位、低层视觉感知、质量描述。  
Q-Bench其实已经做了低层视觉感知、质量描述任务了，ViDA-UGC说数据做的更细，另外更进一步的点是失真定位，我认为思路很好，但是定位用矩形框还是存在问题，毕竟实际失真区域很难用矩形框来对应  
另外一个点是引入了思维链，不是训练一问一答，而是训练多步思考，分了两种任务，整体质量分析任务分成5步（简洁描述、分析失真、找关键失真、分析整体质量、评分），分失真评估任务分成3步（统计失真数量、描述失真4个属性、 总结整体质量影响），效果会比直接问答给结果要好  
基于MLLM微调做IQA，关键还是怎么设计问答对，然后构造对应的数据集  
  
### FG-IQA  
细粒度IQA问题相关的论文单独列一下  
#### （21.6.17国科大 细粒度IQA综述）Fine-Grained Image Quality Assessment A Revisit and Further Thinking  
开新坑的综述，很清晰地解释了为什么要研究FG-IQA问题：看当前IQA模型整体的SROCC/PLCC/KROCC指标似乎已经很高了，但实际上在具体一个失真级别的图像中测试结果非常差，也就是根本区分不出来  
挺有意思的一个工作就是数学仿真了多个级别之间排序全对，但同一级别之内排序很差的情况，发现只要大的级别预测对，就可以获得0.8甚至0.9的总SROCC/PLCC/KROCC，实际同一级别内的正确率可能很低，定义为“统计掩盖细节”现象  
由此也引入了FG-IQA问题，狭义上指的是FR-IQA数据集上同一失真级别的图像能够准确评分并排序，但是我理解广义上其实就是在小的一段质量区间上也可以取得很准确的排序结果，提升IQA分数预测的“精度”  
文中没有很深入说明为什么要做FG-IQA，我的理解是在一些具体的图像压缩/增强任务上，需要有精度很高的IQA评分作为指标。  
文中对于FG-IQA问题怎么做，大致提出的思路是区分特定的应用场景去优化，介绍了图像压缩、图像去噪、图像重定向、色调映射、其他（对比度增强、超分、去雾）等任务上的一些能取得更好FG-IQA结果的一些研究。大致的做法是基于特定的失真数据，提取特定的特征，然后设计特定的模型。（个人感觉区分场景优化比较工程了，做不到通用的优化，不知道往更通用的FG-IQA去做有没有前途）  
还有关于FG-IQA的挑战，说了很多项，我总结下：一是数据集构建问题，很小的质量差异人其实也很难区分，并且主观评价基本只能做FR的了，但设置参考图可能造成bias；二是泛化问题，针对不同失真提取的特征换一种失真一般就不适用了。放到NR-IQA任务上连失真类型、级别都拿不到，那就更难做了。  
### （23.10.3港理工）REQA: Coarse-to-fine assessment of image quality to alleviate the range effect  
难得专门搞FG-IQA模型的文章。可视化了几个模型的预测分数与实际分数的散点图整挺好，不过指标一般，实验做的也比较少，对比的几个baseline都比较旧了，MetaIQA、HyperIQA、GraphIQA。（其中HyperIQA似乎散点图相关性挺好接近y=x，和我测出来不太一致……）  
搞了个range effect的说法，其实也就是FG-IQA在窄区间内做不好，尤其是极低分段和极高分段由于数据量少更明显（这里不严谨，之前测过做等间隔分段是这样，但等频分段就不一定是这样，和数据集分布有关）。  
解决方法还是比较抽象的魔改模型。模型分两路，一路是ResNet50前3层的分层特征作为局部特征，然后不是直接用，而是拼了个17年CVPR Feedback Networks的反馈块，这个结构我看着还挺新鲜的。另一路是ResNet最后一层输出过一层Transformer作为全局特征。处理后的3层的局部特征+1层全局特征拼在一起再过4层MLP得结果。  
FG-IQA分段SROCC比较的时候是按分数段分的5个档比的。  
  
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
  
### （20.5.29德州大学 VIDEVAL）UGC-VQA Benchmarking Blind Video Quality Assessment for User Generated Content.pdf  
融合很多经典视频特征的方法，只有matlab实现  
  
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
  
### （21.7.28上交 TSN-IQA）Task-Specific Normalization for Continual Learning of Blind Image Quality Models  
持续学习，可以连学多个不同的IQA数据集不会串  
TODO  
  
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
  
### （23.4.12NTU PTQE）Blind Video Quality Prediction by Uncovering Human Video Perceptual Representation  
22年TPQI的进一步扩展  
TODO：  
  
### （23.4 NTIRE VQA竞赛）NTIRE 2023 QA of Video Enhancement Challenge  
淘宝团队基于SimpleVQA的模型拿了冠军，在数据增强上做了比较多。TODO：需要借鉴下  
### （23.4.19上交）MD-VQA Multi-Dimensional Quality Assessment for UGC Live Videos  
TODO  
  
### （23.6.26上交 MinimalisticVQA）Analysis of Video Quality Datasets via Design of Minimalistic Video Quality Models  
似乎是对现有VQA数据集有一些分析  
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
  
### （24.4.17快手组织竞赛论文）NTIRE 2024 ChallengeonShort-form UGC Video Quality Assessment: Methods and Results  
可以看到很多魔改模型的思路去借鉴。因为卷的是准确率，所以基本上都是多模型在一通拼，魔改的会很复杂。  
第一名是上交SimpleVQA作者和NTU FastVQA作者的联队，这有点欺负人了，直接把SimpleVQA和FastVQA的结果组合了，还附加了Q-Align和LIQE两个模型的结果一起，4个拼在一块很强。  
但是整体看下来没有什么新的东西，毕竟是刷分的比赛，基本都是流行的这几个主流模型去一通合并，不过也可以看出来FastVQA（含Dover）、SimpleVQA、Q-Align等几个模型是经过检验的模型。  
### （24.5快手 PTM-VQA） PTM-VQA: Efficient Video Quality Assessment Leverage Diverse PreTrained Models from the Wild  
TODO  
  
### （24.8.26上交）LMM-VQA Advancing Video Quality Assessment with Large Multimodal Models  
很强的一篇基于MLLM做VQA的工作，性能乱杀超了Q-Align  
比较创新的点是：  
1. 新设计了视频encoder，搞成双分支一边关注时域（SlowFast）一边关注空域（ViT），然后第一阶段训练投影层做特征对齐（用一些图像描述数据集），第二阶段再VQA任务上指令微调  
2. 一个视频对应的训练的问题对搞成两个，一个是预测连续分数，一个是预测离散级别，效果不错可以直接问连续分数回答了  
但是说开源好像github库没东西  
  
### （24.9.27pub法国南特大学 对齐数据集）A Dataset for Understanding Open UGC Video Datasets  
也是和YouTube合作的  
很有意思的VQA数据集工作，考虑到现有的VQA数据集评分标准不一致，从6个经典数据集（  
YouKu-V1K, YouTube_UGC, LiveVQC, KoNViD-1K, KoNViD-150K, Netflix Public dataset）中各选了一些视频重新评分做对齐比较  
  
### （24.11.6上交）VQA^2: Visual Question Answering for Video Quality Assessment  
大模型做可解释性VQA的很强的工作，做了一个157755个关于视频质量问题问答对的数据集，从而可以训练LLM做视频质量理解任务。  
数据集分为3个子集，一是关于失真分类的问答对，现有合成失真数据集直接转；二是关于质量评分的问答对，现有数据集MOS直接转；三是关于视频质量理解的问答对，靠专家人工注释并通过GPT转换格式（其实主要还是已经有些研究的low-level visual question answering，不过这里数据还会涉及一点美学和AIGC的分析稍微全面一些）。  
模型基于做视觉问答很强的LLaVA-OneVision-Chat-7B训练，一阶段预训练用失真分类数据集1，只训练vision tower和vision projector；二阶段训练就是全参的了，用质量评分数据集2，不过说UGC视频和流媒体视频的评分不一样，分开训练两个模型；三阶段基于做二阶段UGC视频评分模型，继续全参训练，用视频质量理解数据集3。  
最后看VQA性能，二阶段搞出来的UGC评分模型挺强的，再经过三阶段VQA性能有所下降但也基本平齐q-align水平，而且Q-Bench-video上效果不错，超了GPT-4o  
  
### （24.12.24谷歌）An Ensemble Approach to Short-form Video Quality Assessment Using Multimodal LLM  
经典VQA模型集成MLLM zero-shot VQA分数预测，在YouTube-SFV数据集上能涨点，FastVQA、Dover、ModularBVQA都测了，不过涨点非常少，意义有限  
  
涂必超. 图像质量评估综述[EB/OL]. 2018. https://zhuanlan.zhihu.com/p/32553977/.  
YaqiLYU. 全参考图像质量评价方法整理与实用性探讨[EB/OL]. 2018. https://zhuanlan.zhihu.com/p/24804170.  
  
### （25.4.17快手组织挑战赛）NTIRE 2025 Challenge on Short-form UGC Video Quality Assessment and Enhancement: Methods and Results  
25年相比24年，重点放在了两个主题，一是高效，二是超分图片  
高效赛道，第一名SharpMind队的方案很有实际工业应用的借鉴意义，就是先搞尽可能复杂的教师模型，然后蒸馏。教师模型是超全的特征大杂烩，SlowFast、FAST-VQA、LIQE、DeQA、HVS-5M的特征全用了，还加了个Swin-B（这个我不知道是不是并行加的），然后所有的特征全部拼接起来过两层MLP，训练出一个超强的教师模型。  
蒸馏的方法是使用教师模型伪标签标注了30000+的UGC视频，然后让学生模型学。（查了下虽然一般蒸馏是学习教师模型的softmax输出，但是回归模型就一个输出直接学伪标签就行）  
学生模型结构也很经典，直接关键帧随机224x224采样，然后输入Swin-T，4层分层特征全部统一成768拼成3072的输出（不是的过一层dense），然后过两级MLP（第一级3072->384->1得到每个token的分数，然后49->49->1得到总分）  
  
### （25.6.11pub上交 E-VQA NTIRE挑战赛模型）An Empirical Study for Efficient Video Quality Assessment  
NTIRE 2025 高效VQA赛道并列第3名（至少准确率和复杂度都是优于FastVQA的，和第一名SROCC差距也不大0.934 0.926，不过计算量114GFlops还是多了，第一名47GFlops有点夸张，SimpleVQA是1060，Q-Align是12000），写了workshop论文给比较完整的说明，可以借鉴下  
VQA训练流程看做三个步骤：视频预处理、质量感知特征提取和优化策略，文中探讨了每一步的最佳实践  
视频预处理：分析了不同的patch分辨率、关键帧时间采样间隔  
质量感知特征提取：做的经典的双路，关键帧全图resize低分辨率过Swin-T，高分辨率采样patch（用的2D FastVQA） —— 感觉不太对，不应该类似SlowFast低分辨率多帧，高分辨率少帧吗？  
优化策略：分析了不同的损失函数（排序Loss、L1 Loss、L2 Loss、PLCC Loss）。做了知识蒸馏  
确实这里看来这个工作在模型结构上没有做的特别复杂，毕竟高效赛道模型还是比较基础，就是很好的模型调参可以借鉴。当然结果还是比不过第一名的超强知识蒸馏
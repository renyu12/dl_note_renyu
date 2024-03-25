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

## 关于数据集
### IQA数据集
比较经典的4个常用的IQA数据集列在表里  
| 数据集 | 参考图像数 | 失真图像数 | 失真类型数 | 测试人员数 |
| --- | --- | --- | --- | --- |
| TID 2013 | 25 | 3000 | 24 | 971 |
| TID 2008 | 25 | 1700 | 17 | 838 |
| CSIQ | 30 | 866 | 6 | 35 |
| LIVE | 29 | 779 | 5 | 161 |
还有一些时间久的IVC、Toyama、A57、WIQ数据量比较小了  
TODO: 也有一些新的可能还没有完全普及作为通用标准，需要研究下，例如KonIQ-10K, PieAPP, PIPAL, FLIVE, SPAQ, KADID-10k  
#### TID 2013
这个是目前最常用最权威的数据集。都是512\*384的BMP文件。3000张=25张图片\*24种失真类型\*5个失真等级  
TODO: 具体格式  
### VQA数据集
CVD2014, KoVNiD-1k, LIVE-VQC, Youtube-UGC, LSVQ  
DIVIDE-3k（Weisi Lin老师团队考虑审美分数新建的数据集，3634个训练视频，909个验证视频，每个视频都有审美分数、技术分数和整体分数）
TODO  
### PCQA数据集
WPC、SJTU-PQA、M-PCCD、IRPC  
SIAT-PCQD、PointXR，一般点云质量评估是固定观察距离的图片/可旋转视角去评估，这两个是支持6DoF移动观察的
#### WPC
Waterloo Point Cloud，滑铁卢大学发的一个比较大的彩色点云质量评估数据集，对应论文Perceptual Quality Assessment of Colored 3D Point Clouds（2021）  

# 一些研究的思路
模拟人类视觉更关注图片重点内容，显著的区域质量重点处理。ROI (Region of Interest, 感兴趣区域)  
采样降低处理数据开销提升算法性能  
多尺度，结合global和local质量的情况分析  
多种各有优势的网络模型混合  
最小可察觉误差JND，把来自人眼视觉系统特性&心理效应的视觉冗余量化出来，可以指导编码和质量评估  
结合LLM多模态输入图像给出文本评价的能力，进行质量评估  
zero shot方法？  

# 论文整理
## VQA
### （22.6.6 NTU）FAST-VQA: Efficient End-to-end Video Quality Assessment with Fragment Sampling
问题：主要解决了VQA中高清视频数据量过大，计算开销高的问题  
方法：  
* 网格化补丁采样  
把原始视频帧按网格分块，然后每一块里原始分辨率采样一个小补丁，不同帧都在相同位置采样这个小补丁。——这样就采样获取了最细节局部纹理信息  
把所有网格的小补丁拼接在一起组成一个“大补丁”作为输入。——这样就有了个很抽象的全局语义信息（直观感觉有点弱，这都成块了啥也看不出来）  
* FANet网络  
backbone用的是4层注意力层的Swin-T网络，但还要做一点调整。主要问题是大补丁毕竟是硬拼起来，小补丁内Intra-patch像素关联强，小补丁之间cross-patch像素关联弱。所以，一是做注意力计算的时候要有所区分，加一个偏差值隔开不同小补丁；二是不能先池化再非线性回归（这是激活函数吗？）会把小补丁混了，顺序改成先非线性回归再池化。  

想法：这样很小的采样又硬拼在一起，相比更早的方法直观感觉是增加了细节纹理的权重、削弱了全局语义的权重，也能取得很好的效果，应该说明了目前的质量评分基本上还是以局部细节纹理为主吧。  
### （22.11.9 NTU）Exploring Video Quality Assessment on User Generated Contents from Aesthetic And Technical Perspectives
问题：之前的VQA只关注了技术角度的质量评分  
方法：  
* 做了DIVIDE-3k数据集  
每个视频标签是技术评分+审美评分+整体评分  
* 提出了DOVER视频质量评估模型  
就是分了两路分别计算技术评分和审美评分  
TODO：DOVER具体网络设计，我理解技术分应该还是类似Fast-VQA的处理，审美分主要依赖全局语义可以做下采样，但是提了个Cross-scale Regularization没看懂是什么，不同尺度下采样之间比较吗？  
### （23.5.22 NTU）Towards Explainable In-the-Wild Video Quality Assessment: A Database and a Language-Prompted Approach
基于DOVER时搞的DIVIDE-3k数据集，又扩展了评分的维度，本来只有审美评分+技术评分+整体分，又对数据集里4543个视频收集了200w+的文本评价，把这些评价转换成了13个维度的评分。这个数据集确实是个极大的亮点，借助文本语义的信息来解决原本视频质量只有技术评分的问题。  
提出了MaxVQA，这里的Max指的是多维度。  
TODO：网络设计需要再看下，技术分还是用了Fast-VQA，其他语义评分的处理用了做多模态的CLIP  
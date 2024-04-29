本笔记主要基于NTU EE6497 模式识别与深度学习 23-24 第二学期的前半部分模式识别内容，后半部分深度学习笔记汇总至深度学习笔记。  
虽说是模式识别，内容主要侧重于模式识别教材中偏概率统计基础的部分。  
Tay Wee Peng老师前6周模式识别，后面Wang Lipo老师深度学习。  
可以结合coursera YouTube去做点实验，之前相关的本科课程也有点实验，课是理论和实验比较均衡的，想多动手还是要课后下功夫，会提供Python代码，但不会很细致的讲。  
参考教材是机器学习的三大经典教材：  
* PRML：Christopher Bishop, Pattern Recognition and Machine Learning, Springer, 2007.  
* 最牛教材：Kevin P. Murphy, Machine Learning: A Probabilistic Perspective, The MIT Press, 2012.  
* ESL：T. Hastie, R. Tibshirani, and J. Friedman, The Elements of Statistical Learning, Springer, 2009.  
  
  
  
# 1.18 第一课 introduction  
## 概述  
### 机器学习  
监督学习 vs 无监督学习  
### 老师的研究  
老师搞图神经网络，还结合了EM、MCMC算法啥的  
和车企合作搞的无人驾驶目标检测  
## 复习概率论  
### 概率密度函数及其性质  
$p_x(x)$是描述随机变量X概率分布的，也就是X=x的概率。  
离散变量应该叫pmf probability mass function，连续变量是pdf probability density function。本课程的讨论中简单统称为pdf。  
### 联合、边缘概率密度，条件概率  
概率密度到概率$P((X,Y)\in A)=\int_Ap(x,y)\mathrm{d}x\mathrm{d}y$  
联合概率密度->边缘概率密度$p(x)=\int_{-\infty}^\infty p(x,y) \mathrm{d}y$  
条件概率密度$p(y|x)=\frac{p(x,y)}{p(x)}$  
条件概率密度->联合概率密度$p(x,y)=p(x)p(y|x)$，只有在X和Y相互独立的情况下是$p(x,y)=p(x)p(y)$  
贝叶斯定理$p(x|y)=\frac{p(y|x)p(x)}{p(y)}$，后面会用的滚瓜烂熟~  
### 期望和方差（要熟悉多维情况）  
#### 期望&（协）方差  
TODO：参考概率论笔记吧，比较多懒得记了，有空补上  
#### 随机向量、矩阵求导  
TODO：这个好像有点厉害，没有考虑过多维求导问题，研究下对于深度学习很多运算会更清楚一些  
### 参数模型与非参数模型  
这里主要是理解个概念，参数模型假设数据分布遵循特定的数学形式，例如接下来马上要学的各种标准分布、混合分布、马尔科夫链等，可以用MLE、MAP的方法根据训练数据确定模型参数，从而掌握整个模型，这种方法假设其实是很强的（凭什么说数据分布就遵循xxx分布？）  
非参数模型的概念会有点抽象不好理解，定义是不假设数据分布有什么形式，直接从数据中搞一些其他信息用于分析，那其实包含的范围非常广泛，我感觉还是要结合一些具体的例子来理解。反正啥K近邻、随机森林、决策树、支持向量机都可以说是非参数模型，神经网络也可以说是非参数模型。  
### 共轭分布  
还是看贝叶斯定义，后验分布Posterior正比于似然函数Likelihood乘以先验分布Prior  
$P(\theta|D)=\frac{P(D|\theta)P(\theta)}{P(D)}$  
这个式子一定理清楚，后验分布代表已知训练集之后模型最可能的参数分布，根据训练数据获得模型参数是我们需要的结果，但是不好直接求，需要根据似然函数和先验分布去算，分母就是训练数据的分布是已知的但是不重要，当做一个归一化的系数。似然函数理解为确定了模型参数之后采样训练数据的分布，后面学最大似然估计MLE就是认为当前采样的训练数据就是最符合模型参数的分布；先验分布是我们根据知识，认为模型应该符合的分布，这个其实也难得到，有的学者认为贝叶斯模型有点扯就是因为需要有比较好的先验分布才有比较好的模型结果。  
扯了这些概念，共轭分布就很好理解了。如果先验分布和后验分布的分布形式相同（例如都是Beta分布/高斯分布），就称为共轭分布conjugate distribution。此时先验分布称为似然函数的共轭先验conjugate prior（这里区分下概念，共轭分布是先验x后验，共轭先验是先验x似然函数，看起来如果一个成立应该另外一个也成立）。  
那肯定能想到满足共轭分布一定有什么好的特性可以用。简单而言就是满足共轭分布，有新加入的数据的时候，可以直接通过计算新加入数据更新模型参数（就把当前的情况作为“新的”先验分布即可）。反之不满足共轭分布，一旦有新加入的数据，必须加上老数据一起全部重新计算很麻烦。所以如果是共轭分布将大大减少模型更新的计算量。  
## 重要的标准分布  
### 均匀分布和指数分布  
#### 均匀分布  
$X\sim Unif(a,b)$ $Unif(x|a,b)=\begin{cases} \frac{1}{b-a} \text{ ,if } a\le x \le b\\0 \text{ ,otherwise}\end{cases}$  
$E(X)=\frac{a+b}{2}, Var(X)=\frac{(b-a)^2}{12}$  
#### 指数分布  
$X\sim Exp(\lambda)$ $Exp(\lambda)=\begin{cases} \lambda e^{-\lambda x} \text{ ,if } x \ge 0\\0 \text{ ,otherwise}\end{cases}$  
$E(X)=\frac{1}{\lambda}, Var(X)=\frac{1}{\lambda^2}$  
$\lambda_{MLE}=\frac{n}{\Sigma_{i=1}^nx_i},\text{ log likelihood}=n\log \lambda -\lambda\Sigma_{i=1}^nx_i$  
### 伯努利分布、二项分布、Beta分布  
#### 伯努利分布  
$X\sim Bern(\theta)$就是两点分布/0-1分布，最简单的那个，$P(x)=\theta^x(1-\theta)^{(1-x)}$，即x为1的概率为θ，x为0的概率为1-θ  
期望是θ，方差是θ(1-θ)  
#### 二项分布  
$X\sim Bin(\theta)$，就是重复伯努利实验得到1的次数，得1和0的概率都累乘，还要考虑下哪几次得1做个组合，$Bin(\theta)=C_n^x\theta^x(1-\theta)^{n-x}$  
$E(X)=n\theta, Var(X)=n\theta(1-\theta)$  
#### Beta分布  
$X\sim Beta(a,b)$，这个分布看起来形式好像很复杂其实挺简单的，就是代公式算出来就行。$P(x|a,b)=\frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}x^{a+1}(1-x)^{b-1}$  
其中$\Gamma(z)=\int_0^\infty u^{z-1}e^{-u}\mathrm{d}u=(z-1)!$，所以其实很简单  
$E(X)=\frac{a}{a+b}, Var(X)=\frac{ab}{(a+b)^2(a+b+1)}$  
### 分类分布、多项式分布、Dirichlet狄利克雷分布  
分类分布和伯努利分布类似，只是有K类不止2类了。  
多项式分布和二项分布比较有关联，把每次只有2个结果的独立实验扩展成多个了（也可以看做是分类分布的扩展）TODO：没太看懂……指的是n次独立实验中成功k次的概率分布，k=2就是二项分布。  
Dirichlet分布是Beta分布的多维推广，和伯努利分布、二项分布的关系一样。  
### 正态/高斯分布  
$X\sim N(\mu,\sigma^2)$  
公式都比较熟了，直接记log likelihood$=\Sigma_{i=1}^n\log\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x_i-\mu)^2}{2\sigma^2}}=\sum_{i=1}^n(-\frac{1}{2}\log(2\pi\sigma^2)-\frac{(x_i-\mu)^2}{2\sigma^2})=-\frac{n}{2}\log (2\pi\sigma^2)-\frac{1}{2\sigma^2}\Sigma_{i=1}^n(x_i-\mu)^2$  
#### 高斯分布线性组合  
首先假设$X\sim N(\mu,\sigma^2)$，则  
$aX\sim N(a\mu,a^2\sigma^2)$  
$X+c\sim N(\mu+c,\sigma^2)$  
再假设$Z\sim N(0,1)$，则  
$X=\sigma Z+\mu \sim N(\mu,\sigma^2)$  
再假设$Y\sim N(\xi,v^2)$和X相互独立，则  
$X+Y\sim N(\mu+\xi,\sigma^2+v^2)$  
#### 多维高斯分布  
麻烦的就在这里，现在X是n维向量了，Z是k维向量，每一维都是标准高斯分布N(0,1)，然后均值μ也是k维向量，那么X可以表示为$X=AZ+\mu$。这里有个复杂的地方就是A是n行k列的矩阵把k个高斯分布组合成n维的随机向量，这代表了不同维之间可能有关联，最特殊的情况才是各维都是独立的一维高斯分布，协方差矩阵只有对角线。  
X可以叫做联合高斯分布jointly Gaussian也可以叫做多维高斯分布Multivariate Gaussian，是一个东西。  
$E[X]=E[AZ+\mu]=AE[Z]+\mu=\mu$，即均值还是看μ  
$cov(X)=E[(X-\mu)(X-\mu)^T]=E[AZZ^TA^T]=AE[ZZ^T]A^T=AA^T$，协方差矩阵直接是A向量乘自己的转置。  
多维高斯分布的公式$N(\mu,\Sigma)=\frac{1}{(2\pi)^\frac{K}{2}(|\Sigma|)^\frac{1}{2}}e^{-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)}$  
协方差矩阵应该比较熟了，对角线是方差，其他是$cov(X_i,X_j)=\rho_{X_i,X_j}\sigma_X\sigma_Y$  
  
# 1.25 第二课 Bayesian Inference  
认真看看老师的课件、代码、手写笔记、习题讲解都很全  
写pdf的时候$p(X=3.24),p_X(3.24)$都可以  
## 贝叶斯模型  
infer的意思是从一个随机变量X推导出另一个随机变量Y，做分类、回归、预测，贝叶斯推理可能含有模型参数θ也是随机变量  
似然模型likelihood model $p(x|\theta)$  
例如现在要用一个二次曲线$y=a_0+a_1 x+a_2 x^2$去拟合训练数据，如何确认系数的最佳值？有很多种方式。贝叶斯方式就是考虑已知一些x y的情况下，分析$p(\theta|x,y)=\frac{p(x,y|\theta)p(\theta)}{p(x,y)}$。最终就可以通过x和θ去预测y，即$p(y|x,\theta)$，这是discriminative model。  
假设是无监督任务，给定训练数据集$D=\{x_1,x_2,...,x_n\}$。如果X服从Bern分布，各个值iid，θ[0,1]，则$p(D|\theta)=\theta^{N_1}(1-\theta)^{N_0}$，因为独立同分布就是累乘下，N1和N0是x取值为1/0的个数。如果X服从指数分布或者X服从高斯分布，确认对应的分布参数θ，也可以得到训练数据的pdf：$p(D|\theta)$  
## MLE  
最大似然估计 maximum Likelihood Estimate  
基本思路是最大化训练数据的pdf $\theta_{ML}=arg \max p(D|\theta)$，即认为训练数据分布就是符合整体数据分布的典型  
数学上如何操作？首先做$p(D|\theta)$的最大值可以做个log转化为求$\log p(D|\theta)$的最大值，这样一个独立同分布概率累乘的结果就可以变成一个累加的结果$\sum_{i=1}^n \log p(x_i|\theta)$，求其对θ的偏导，可以化为每个log pdf求偏导的累加（第一周有课件介绍相关的数学基础）。例如上面的Bern分布pdf，求导后是$\frac{N_1}{\theta}-\frac{N_0}{1-\theta}$，计算其等于0可以得到极值点求出最优的θ。  
同理，上面的指数分布，高斯分布的训练数据pdf也都是可以取log然后求偏导从而找到最优模型参数的。  
所以实际算题很简单，就是把训练数据各自的分布累乘，取log，然后求导去求极值点确认最大值点的θ就是答案。  
### 补充Kullback-Leibler Divergence  
就是KL散度，衡量两个分布之间的差异，是MLE可行的理论基础，因为可以最小化KL散度  
### 线性回归  
然后就可以学习下最经典的线性回归方法，认为模型是$y=w^Tx+\epsilon$，$\epsilon$服从均值为0的高斯分布，所以$w^Tx$确定均值（TODO，这里方程式有的变量名不会敲）  
pdf取log得到 $\log p(y|\phi(x),w)=\sum_{i=1}^n\log N(y_i|w^Tx_i,\sigma^2)$，可以化为范数形式$-\frac{1}{2\sigma^2}||\Phi w-y||^2+const$  
然后说模型的非线性关系使用basic function的expansion，没懂，应该是有二次以上的项。感觉老师这里给的basic function的定义非常宽泛，用logx啥的都行，甚至$x_1x_2$这样的项也认为是的，有点神奇。  
求权重用MLE，推导也是不太明白，用到了之前提到的转置和求偏导啥的公式……反正也是求偏导等于0得到极大值点的$w_{ML}=(\Phi^T\Phi)^{-1}\Phi^Ty$，然后就可以用$y=w_{WL}^T \phi(x)+\epsilon$来预测y了。可以用RSS（Residual sum of squares）来分析误差，$RSS=\sum_{i=1}^n(y_i-\hat{y_i})^2$，还有RMSE、R^2，R^2越大预测越好，直接用均值做预测的话R^2为0，但其实可以为负值。  
看看多项式回归的代码，老师推荐多上github找找代码学习。代码写的很全直接做了用1-20次多项式去拟合训练数据然后在验证集上验证的效果，可以看出多项式次数越高对训练集拟合越好，但是实际数据可能是二次的，后面过拟合导致二项以后验证集效果越来越差。  
感觉老师这里给的线性回归的介绍相当粗略~  
## MAP  
最大后验估计Maximum A Posteriori Estimate，对比MLE相当于扩展，还融合了预估计量的先验分布信息，也就是考虑了p(θ)本身，而不是直接通过pdf最大去计算θ，因为实际要最大化的是$p(\theta|D)=\frac{p(D|\theta)p(\theta)}{p(D)}$，也就是根据观测的数据D反推出分布模型的参数θ，分母不重要忽略，直接分子都考虑了。  
考虑了$log(p(D|\theta)p(\theta))=log p(D|\theta) + log p(\theta)$  
以分类问题做例子，例如图像识别分辨猫和狗，目标是最小化分类错误率。给定x，则分类错误率定义为$\sum_{y != \delta(x)}P(Y=y|x)=1-P(Y=\delta(x)|x)$，然后用MAP做。  
考虑用Naive Bayes朴素贝叶斯去做MNIST手写数字识别（当然是不好的，只是学方法），拆分开每个像素去看？（老师没有细讲朴素贝叶斯，研究了下朴素贝叶斯简化的思路就是认为各个特征之间是独立的，然后就把各个特征的概率直接乘起来做似然函数了，所以这里应该就是忽略像素之间的关联性不用考虑条件概率）反正也能实现84.3%的准确率，非常简单，可以去学习下代码。  
  
# 2.1 第三课 Mixture Models and EM  
## Mixture Models 混合模型  
已经学了一些有用的模型了，伯努利、指数、高斯、MLP、CNN、RNN、GNN……一般做法是具体应用选具体一个合适的模型，可以用AIC、BIC（啥？）这样的指标来选。然后根据训练集数据确定模型的参数，找到拟合训练集最佳的参数值，方法是用MLE（asymptotially corred，啥？）、MAP（用到先验概率）、定义损失函数找最小值。然后确定模型参数就可以用模型去预测，在测试集验证。  
混合模型的思路就是再组合学过的一些基础的模型，一些问题不是一个基础模型就能很好拟合的。  
给了个用100个高斯分布去拟合一个兔子点云的Git项目，有点奇怪。  
混合模型除了每个分布的pdf，还需要考虑一个点是属于哪一个分布，我们编号K个组成的分布获得1到K的索引，那又引入了一个变量叫做latent variable，常用z表示，对应着一个点来自第z个分布。此时$p(x|\theta)=\sum_{k=1}^Kp(x,z=k|\theta)=\sum_{k=1}^K\pi[k]p(x|\eta_k)$，此时参数θ就是$\pi,\eta_k$两个，π[k]是第k个分布所占的权重（注意所有权重π累加和为1，这样子各个分布的概率密度才累加为1，不然就不对劲了），$\eta_k$就是第k个分布的pdf参数，其pdf为$p(x|\eta_k)$。  
总结一下混合模型的形式其实非常简单，就是直接加权平均不同的分布。  
### GMM Gaussian Mixture Model 高斯混合模型  
很简单，知道了混合模型的定义，GMM就是具体模型用高斯。  
$p(x|\theta)=\sum_{k=1}^K\pi[k]N(x,\mu_k,\Sigma_k)$  
这里需要补充下多维高斯分布参数不是均值和方差了，而是均值向量（n维）和协方差矩阵大Sigma（n\*n矩阵表示n维变量之间的相关度，对角线上是每个维度变量自己的方差）  
关于优化的问题不了解老师上传了一个优化问题回顾的视频，条件极值用拉格朗日乘数法要会。现在做GMM的训练也需要理解。  
#### 补充：GMM期望和方差推导  
从练习题来的记录下  
$E[X]=E[E[X|Z]]=\sum_{k=1}^K\pi[k]E[X|Z=k]=\sum_{k=1}^K\pi[k]\mu_k$  
$Cov(X)=E[(X-EX)(X-EX)^T]=E[XX^T]-E[X]E[X]^T=\sum_{k=1}^K\pi[k]E[XX^T|Z=k]-E[X]E[X]^T=\sum_{k=1}^K\pi[k](\Sigma_k+\mu_k\mu_k^T)-E[X]E[X]^T$  
这里需要补充证明下$E[XX^T|Z=k]=\Sigma_k+\mu_k\mu_k^T$，这个乍一看不直观，其实还是协方差矩阵不那么熟，把协方差矩阵拆开就行$\Sigma_k=E[(X-\mu_k)(X-\mu_k)^T|Z=k]=E[XX^T|Z=k]-E[\mu_kX^T|Z=k]-E[X\mu_k^T|Z=k]+\mu_k\mu_k^T=E[XX^T|Z=k]-\mu_k\mu_k^T-\mu_k\mu_k^T+\mu_k\mu_k^T=E[XX^T|Z=k]-\mu_k\mu_k^T$，得证  
#### 尝试GMM MLE  
训练还是MLE问题，有了数据x1到xn，独立同分布，$p(x_i|\theta)=\sum_{k=1}^K\pi[k]p_k(x_i|\theta)$  
做对数，最大化这里的似然函数  
$\log p(x_1,...,x_n|\theta)=\sum_{i=1}^N\log\sum_{k=1}^K\pi[k]p(x_i|\theta)$  
#####  Singularities 奇点问题  
情况要比单高斯更复杂了，因为有多个高斯分布，如果有一类只有1个点$x_i$，发现取$\mu_k=x_i, \sigma_k \to 0$，直接MLE就趋于无穷了，看起来很好但其实是没法算的。解决方法是reset均值为随机值，reset方差，TODO：咋reset啊？  
#####  Unidentifiability 不唯一问题  
理想情况我们做MLE的时候希望只要参数不同$\theta_1 \ne \theta_2$，概率分布$p(x|\theta)$就不同，这样就好确定唯一的一组参数。但是GMM下，肯定存在多个不同的参数组合能实现一样的最优pdf（想一想简单permute各组高斯分布序号k就行了）。有很多种匹配分布index和具体数据点的方法。  
但这个问题给搜索引入了复杂性，倒是不影响最优点的结果，这说明了全局最优点不是唯一的，但我们只要找到一个即可。  
##### Optimization 优化问题  
只有x不完整的数据格式，似然函数$\sum_{i=1}^n\log\sum_{k=1}^K\pi [k]p_k(x_i|\theta)$，因为log里面还有累加（不是累乘了），这个无法拆分为log累加形式，是不好解的，求导后形式会很复杂。  
解决方法是引入latent variables z1到zn一起算，$(x_i,z_i)$作为完整的数据格式，形式就可以调整下，log里面不用累加所有k类的$\pi[k]$了，已经确定了是$z_i$类，所以上式$\pi[k]$替换为条件概率$p(z_i|\theta)$，直接外面累加每一个点就可以了，$\sum_{i=1}^n(\log p(z_i|\theta)+\log p_{z_i}(x_i|\theta))$  
然后是最大化这个likelihood函数，不好做，引入EM算法（下面EM算法部分会进一步分析）。  
形象地想一下，知道数据集中的一点x，其实是不那么容易确定它是属于哪个高斯分布的，因为高斯分布毕竟是长尾的，理论上任何一个都是可能的。已知模型参数去预测一个点的分布就是要按权重加和所有高斯分布。而如果告诉你一个点是输入z=2的第二个高斯分布，那这个时候情况就清晰了，所以我们想要拿完整数据y(x,z)。  
## EM 算法  
### EM算法作用和思想  
#### 分析隐变量MLE/MAP的困难  
首先要清楚EM算法是一个很通用的“在模型概率依赖无法观测的隐变量时”仍然可以做MLE/MAP获得模型参数最佳估计的算法，不只是GMM模型可以用。可以理解为含隐变量时的MLE/MAP。  
为什么有了隐变量就不好做MLE/MAP？很简单就以GMM为例，我们要估计多个高斯分布的参数，如果有完整的观测结果$y(x_i,z_i)$，即样本点我们又知道值，又知道是属于哪一个高斯分布的，那非常好做。  
这里光说其实还是不好理解，看看似然函数公式可能好一些。TODO：我对这里的公式还是理解不太深，课件上直接给的对数似然函数但是没有给推导……也没有在拿到完整数据时的例题……  
$P(y|\theta)=P((x,z)|\theta)=\sum_{i=1}^nP(z_i=k|\theta)P(x_i|z_i=k,\theta)=\sum_{i=1}^nP(z_i=k|\theta)N(x_i|\mu_k,\Sigma_k)$  
TODO：$P(z_i=k|\theta)$对应的$\pi[k]$怎么估计的我不是很确定……review1中有个例题是给定了固定的权重值。不管那么多了，自己查了下，权重值$\pi[k]$就直接按照每一类样本的占比去估计即可。然后就可以分开做，在每一类中估计各个高斯分布的均值$\hat{\mu_k}=\frac{1}{n}\Sigma_{i:z_i=k}x_i$和$\hat{\Sigma_k}=\frac{1}{n}\Sigma_{i:z_i=k}(x_i-\hat{\mu_k})(x_i-\hat{\mu_k})^T$  
但问题是实际观察到的数据集是不完整的，只有x，但不知道这个x点是属于第z个高斯分布的。这个时候再去看似然函数，那不知道k的情况下，用π[k]累加也太痛苦了。  
#### EM算法思想  
我们试着最大化如下另一个式子：  
$E_{p(y|x,\hat{\theta}})[\log p(y|\theta)|x,\hat{\theta}]$  
这个式子的设计其实是很绝的，说起来这像是拿到完整数据时的对数似然函数的期望，但是这是个条件概率，在两方面的条件下，一方面是给的的观测数据X，一方面是猜测的参数。TODO：我还是理解不清为什么设计成这样就行了……可能需要实际计算下才能搞清楚。  
这里引入了另一个参数$\hat{\theta}$是对于θ的猜测，因为$E_Y[f(x,y)]=g(x)$，所以这样子就不需要知道y，只要分析θ就行了。EM算法的流程如下：  
* 1.初始化一个猜测的$\theta^{(0)}$  
* 2.E step: 对于第m+1次迭代，估计$Q(\theta|\theta^{(m)})=E_{p(y|x,\theta^{(m)})}[\log p(y|\theta)|x,\theta^{(m)}]=\int\log p(y|\theta)\cdot p(y|x,\theta^{(m)})\mathrm{d}y$  
* 3.M step: 通过最大化Q，找到第m+1次的对θ的猜测，公式为$\theta^{(m+1)}=arg\max_{\theta \in \Theta}Q(\theta|\theta^{(m)})$  
* 4.重复EM步骤直到达到收敛的标准（肯定会收敛的）  
  
EM算法的思想是非常聪明的，就是分了两步，第一步固定猜测的参数，就可以计算Q函数，相当于是对未知数据的估计；第二步固定Q函数，也就是认为未知数据的分布已知，然后去找让Q函数最大的参数值，调整模型参数。以此反复迭代可以逼近已知完整数据时的结果。  
### EM for GMM  
来试一下用EM算GMM，老师说学习这个做个更优秀的人，估计一生也不会推导几次这个。  
我去，式子有点长不好记了，TODO，有机会回头手抄一遍吧。  
#### E step Q函数公式推导  
$Q(\theta|\theta^{(m)}=E_{p(y|x,\theta^{(m)})}[\sum_{i=1}^n\log p(x_i,z_i|\theta)|x,\theta^{(m)}]$  
$=\sum_{i=1}^nE_{\theta^{(m)}}[xxx]$  
可以转化为各个期望E按权重的加和  
$=\sum_{i=1}^n\sum_{k=1}^Kp(z_i=k|\theta^{(m)})E_{\theta^{(m)}}[\log p(x_i,z_i=k|\theta)]$  
关注后面的期望值，把期望里的条件概率拆一下  
$=xxx E_{\theta^{(m)}}[\log p(z_i=k|\theta)+\log p(x_i|z_i=k,\theta)]$  
$=\sum_{i=1}^n\sum_{k=1}^Kp(z_i=k|x_i,\theta^{(m)})[\log\pi[k]+\log N(x_i|\mu_k,\sum_k)]$  
简化下，定义  
$r_{ik}^{(m)}=p(z_i=k|x_i,\theta^{(m)})=\frac{p(z_i=k|\theta^{(m)})p(x_i|z_i=k,\theta^{(m)})}{p(x_i|\theta^{(m)})}$  
$=\frac{\pi^{(m)}[k]N(x_i|\mu_k^{(m)},\Sigma_k^{(m)})}{\sum_{k'} \pi^{(m)}[k']N(x_i|\mu_{k'}^{(m)},\Sigma_{k'}^{(m)})}$  
注意其中  
$N(x|\mu,\sum)$正比于  
反正可以最后得到  
$Q(\theta|\theta^{(m)})=\sum_{i=1}^n\sum_{k=1}^Kr_{ik}^{(m)}\log\pi [k]+\sum_{i=1}^n\sum_{k=1}^Kr_{ik}^{(m)}\log N(x_i|\mu_k,\Sigma_k)$的复杂式子  
观察下系数$r_{ik}^{(m)}$会发现这里对同一类样本点n个累加，再把高斯分布公式展开，还可以再简化形式，设$n_k^{(m)}=\sum_{i=1}^nr_{ik}^{(m)}$，则Q函数化为：  
$Q(\theta|\theta^{(m)})=\sum_{k=1}^Kn_{k}^{(m)}\log\pi [k]+\sum_{k=1}^Kn_{k}^{(m)}\log|\Sigma_k| -\frac{1}{2}\sum_{i=1}^n\sum_{k=1}^Kr_{ik}^{(m)}(x_i-\mu_k)^T\Sigma_k^{-1}(x_i-\mu_k)$  
从而完成E step  
#### M step 最大化Q函数推导  
接着做M step更新$\theta^{(m)}$，也就是$\pi^{(m)}[k],\mu_k^{(m)},\Sigma_k^{(m)}$  
这里的原理也就是Q函数对各个待估计的参数求导等于0，依次计算出$\theta^{(m)}$，也就是$\pi^{(m)}[k],\mu_k^{(m)},\Sigma_k^{(m)}$。计算过程有点麻烦我也略过了，可以记个结论。  
$\pi^{(m+1)}[k]=\frac{n_k^{(m)}}{n}$  
$\mu_k^{(m+1)}=\frac{1}{n_k^{(m)}}\sum_{i=1}^nr_{ik}^{(m)}x_i$  
$\Sigma_k^{(m+1)}=\frac{1}{n_k^{(m)}}\sum_{i=1}^nr_{ik}^{(m)}(x_i-\mu_k^{(m+1)})(x_i-\mu_k^{(m+1)})^T$  
#### EM算法in GMM计算总结  
总结一下，先搞初始值$\pi^{(0)},\mu_k^{(0)},\Sigma_k^{(0)}$，然后重复E step和M step，直到$L^{(m+1)}-L^{(m)} \lt \epsilon$的时候停止迭代（判断收敛还是用的似然函数而不是Q）  
这里有对应的python代码例子，git上也可以找到，去看看怎么跑的。代码中r是记为responsibility，E step和M step代码都封装了函数直接调用就行，还是步骤很直接的。推导过程老师也承认确实是比较复杂，但是理解基本步骤就不影响调用。  
### EM for MAP  
MAP也可以用EM算法来做。也是认为完整数据是y，现在只知道x，需要估计θ。  
实际步骤和MLE的EM算法差别不大，只是在M步骤中，最大化Q函数变为最大化Q+logP(θ)  
## K-means  
无监督算法，聚类。  
听起来其实和GMM的目标是有点像的，其实可以视为在$\Sigma_k=\sigma^2I,\pi[k]=1/K$固定的情况的EM for GMM。现在就只需要分析$\mu_k$了，非常简化。  
用EM做，E step中$r_{ik}^{(m)}$可以对于最近的类固定为1其他为0（称为hard EM，$k_i=arg\min_k||x_i-\mu_k^{(m)}||^2$），得到$Q(\theta|\theta^{(m)})$的式子也简单一些。M step中得到唯一要确定的参数$\mu_k^{(m+1)}$  
在sklearn.cluster聚类算法库里可以找到KMeans，很好调用  
对于混合模型而言有点不好的地方是你需要提前确认有多少类模型组合，KMeans也需要给出聚几类  
关于Kmeans的健壮性还是挺好的，即使初始化条件啥的比较怪，基本也是结果一致的。还有Kmeans++，可以一开始指派聚类中心  
  
# 补充材料 优化问题回顾  
## 函数极值问题  
### 一维  
可导函数求导分析单调性，很简单  
### 多维  
可导函数，求梯度（即各方向偏导都为0）为0的点。这里还涉及到要二阶偏导连续的问题，因为偏导只代表坐标轴方向上是个极值点  
## 函数条件极值问题  
等式约束 拉格朗日乘数法  
不等式约束 拉格朗日乘数法+KTT条件  
  
# 2.8 第四课 Markov Model and HMM  
  
一般的机器学习流程：  
一.选择模型。目前已经学习了一些模型了。概率模型：标准分布（Bern、Exp、Norm）、混合模型、马尔科夫模型/隐式马尔科夫模型。后面还有loss模型就是神经网络那些。  
二.确定参数。不同模型有不同的参数。  
三.infer模型参数。根据训练集数据infer出最优的模型参数，一般方法有MLE、MAP、EM、Baum-Welch、采样、MCMC、最小化loss（如随机梯度下降）  
四.使用模型推理。现在是inference time，可以应用训练的模型，然后在测试集上预测，评估模型的效果。  
  
## 马尔科夫模型  
### 动机  
语音识别：已经有过去的部分句子内容，分析最新说的一个词是什么  
文本生成：已经输入了一些内容，自动补全后面的  
总之就是根据已有的状态信息预测下一个状态  
  
### 基本概念  
考虑离散的随机变量序列x_0,x_1,...（例如NLP任务，时间序列模型等）  
每一个$x_t \in \text{state space} {1,2,...,M}$，即x取值是在状态空间中的是有限的。  
简单情况下马尔科夫模型的性质是$p(x_t|x_1,...,x_{t-1})=p(x_t|x_{t-1})$，也就是只有最近时刻的历史信息有影响，并不是所有的历史信息都有影响。  
$T(i,j)=p_{x_i|x_{t-1}}(j|i)=p(x_t=j|x_{t-1}=i)$称为transition probability转移概率，代表从i状态转移到j状态的概率。不同的i和j就可以组成一个M\*M的矩阵。也有更复杂的情况转移概率还随着时间变化，我们不考虑，就考虑这里简单的homogeneous的MC（马尔科夫链）  
#### 转移概率矩阵  
行数对应转移的起点i，列数对应转移的目标j。  
每一行T的和都是1，$\sum_{y=1}^MT(i,j)=\sum_{y=1}^Mp_{x_t|x_{t-1}}(j|i)=1$。很明显就是i状态转移到所有状态的概率和为1，当然列就不是了。  
T就是一个row stochastic matrix行随机矩阵  
再进一步分析，已知x_0的pdf是p_0，那x_1的pdf p_1怎么算？咋一看并没有那么直观好算。其实$p_1(x_1)=\sum_{x_0=1}^M p(x_1,x_0)=\sum_{x_0=1}^M p(x_1|x_0)p_0(x_0)=(p_0T)(x_1)$，所以结论很简单，$p1=p_0T$，那么$p_t=p_0T^t$。其中$p_t$的形式是行向量$[p_t(1),...,p_t(M)]$，对应所有状态的离散概率分布pmf。  
  
### 应用：语言模型  
先说早一点的统计语言模型，要学习词序列的概率分布然后预测下一个词。所以就可做句子补全，例如google搜索框的搜索词预测，按可能的下一个词概率排序。这种东西还是在用的，毕竟简单可靠。还可以做数据压缩，用短一些的码字去代表更常见的字符串（这个也算是马尔科夫吗？）还可以做文本分类。  
此时的状态空间会有点大，就是在一个语言中的全部词汇，例如全部英语词汇。  
Unigram Model：$p(x_t=x)$，不考虑历史信息预测词汇  
Bigram Model: $p(x_t|x_{t-1})$，按上一个词去预测下一个  
n-gram Model: $p(x_t|x_{t-1},...,x_{t-n+1})$，按前n-1个词去预测下一个  
  
### 应用：PageRank  
网站被其他权威的网站链接了，就认为这个网站的权威性比较高  
$\pi_i$是网站i的得分  
$\pi_i=\sum_j T(j,i)\pi_j$，由其他所有网站转移而来  
  
### MLE  
现在可以用MLE，根据训练数据来估计π和T了  
考虑一个训练数据(x_0,...x_t，有$p(x_0,...,x_t|\pi, T)=\pi(x_0)T(x_0,x_1)T(x_1,x_2)...T(x_{t-1},x_t)$，可以取log得到$\log \pi(x_0)+\sum_{i=1}^t\log T(x_{i-1},x_i)$  
假设是观察训练数据量为n的训练集D={x_1,...,x_n}，然后一条训练数据就是$x_i=\{x_{i,0},x_{i,1},...x_{i,t_i}\}$，长度是$t_i+1$的一个序列，因为每条数据可能长度不相同，所以是t_i而不是定值  
写出对数似然函数$\log p(D|\pi,T)=\sum_{k=1}^n \log p(x_k|\pi,T)=\sum_{k=1}^n(\log \pi(x_{k0})+\sum_{i=1}^{t_k} \log T(x_{k,i-1},x_{k,i}))$  
括号拆开累加，前者就是在所有训练数据中起始为x的项，后者就是在所有训练数据中找连续的两项（TODO，这里没跟上）  
$=\sum_{x=1}^MN_x\log\pi(x)+\sum_{x=1}^M\sum_{y=1}^MN_{xy}\log T(x,y)$  
其中$N_x=\sum_{k=1}^n 1\{x_{k0}=x\}$，$N_{xy}=\sum_{i=1}^n\sum_{t=1}^{t_i}1\{x_{i,t-1}=x,x_{i,t}=y\}$  
光看这个式子可能还不是那么好理解，来个例子，例如现在就ABC三个状态，来了个数据(BAC)，似然函数应该是$\pi(B)T(B,A)T(A,C)$这几个概率连乘，$N_x$对应B，$N_{xy}$对应B->A和A->C，其他项都是0。  
有了似然函数求导为0得到估计值的计算老师说自己回去练，反正直接给出了$\hat{\pi}(x)$和$\hat{T}(x,y)$  
存在的问题及时如果一些状态没有出现过的话，就不能预测出来，这也是一种过拟合，所以需要很大的训练数据集才能效果比较好  
  
## HMM Hidden Markov model  
### 概念  
离散状态的马尔科夫链含有隐含状态或者隐变量$z_i \in \{1,...,M\}$，时间t还是0,1,...，还有初始的pdf π以及转移矩阵T。  
$x_t$不是由$x_{t-1}$直接推出了，而是$x_t$和隐变量$z_t$相关，$z_t$可以由$z_{t-1}$推出，也就是z是马尔科夫链。  
引入了emission probabilities（好像中文叫发射概率，有点难听）的概念，区分转移概率是在马尔科夫链上的z每次换状态的概率（z->z），而发射概率是对于指定的z，x的概率分布（也就是在特定z下的x的条件概率，从数学角度描述“可观察x与隐变量z相关”就是z是决定x分布的参数，z->x）。发射概率定义是$p(x_t|z_t)=p(x_t|\phi_{z_t})$，这里$\phi=(\phi_1,...,\phi_M)$是隐变量模型的参数。例如（做题的时候）emission probability常用高斯分布啥的，即不同的z对应不同的高斯分布，那么$\phi1=(\mu_1,\Sigma_1)$。  
仔细分析下模型参数$\theta=(\pi,T,\phi_1,...,\phi_M)$，M就是z状态空间中可能的状态数，所以π是M\*1的向量对应不同状态z的初始pdf，T是M\*M的矩阵对应不同状态z的转移概率。  
  
### 应用  
语音识别，x可以看做是语音信号提取的特征，可以直接观察，z是说的词不能直接观察但是符合马尔科夫链，我们认为$p(z_t|z_{t-1})$的转移概率代表了语言模型，$p(x_t|z_t)$是emission probability代表了语音模型。  
行文识别：认为x是图像/视频的一些提取特征，可以直接观察，z是行为类型，不可直接观察但是符合马尔科夫链。  
基因匹配：认为x是核酸（ACGT），z是是否在特定的基因组  
  
### EM for HMM 即Baum-Welch算法  
#### 理想情况：已知隐变量直接MLE  
有隐变量就得EM  
有训练集D包含一串x_1到x_n，每个的形式为$x_i=\{x_{i,0},x_{i,1},...,x_{i,t_i}\}$  
观察隐状态z_i，如果有完整的数据包含$(x_i,z_i)$就会好做很多，可以得到对数似然函数形式的后验概率，从而可以正常做MLE来估计参数$\theta=(\pi,T,\phi)$，也就是参数多一点原理还是一样的。  
老师推导了下这个后验概率，但是形式有点长……没有跟上，TODO可以回去推下  
$\log p(\hat{D}|\theta)=\sum_{i=1}^n\log p(x_i,z_i|\theta)=\sum_{i=1}^n\log(p(z_{i0}|\theta)p(x_{i0}|z_{i0})p(z_{i1}|z_{i0})p(x_{i1}|z_{i1})...)=xxx=\sum_{i=1}^n\log \pi(z_{i,0})+\sum_{i=1}^n\sum_{t=1}^{t_i}\log T(z_{i,t-1},z_{i,t})+\sum_{i=1}^n\sum_{t=0}^{t_i}\log p(x_{i,t}|\phi_{z_{i,t}})$  
看着式子有点麻烦，看清就是含有$\pi,T,\phi$这三个参数的式子就好。  
#### 实际情况：未知隐变量老老实实EM  
但实际上没有$z_{i,t}$，所以要做EM算法  
这个推导过程比EM for GMM还长，老师都不想推了，直接作为supplementary，当然老师还是希望大家推一下to be a better person  
E step就是得到$\theta$的估计，其中一些中间变量$\gamma_{i,t}(z),\xi_{i,t}(z,z')$可以用forward-backward算法计算，也是比较复杂放到supplementary里了，老师说这里做的挺漂亮的  
M step可以迭代得到π和T参数的估计，还有emission probability参数$\phi$也要估计，如果是用的高斯分布，那就是得到$\mu_z,\Sigma_z$的估计值（这个点后面在Quiz里还考了，就是问BW算法估计的是什么参数，乍一听有点迷，但是看看后验概率公式的形式就会知道共有三类参数$\pi,T,\phi(\mu，\Sigma)$都是要顾及的）  
  
### 不同的推理方式inference in HMMs  
Filtering方式。infer隐状态。只用到过去的数据，也就是Forward算法  
Smoothing方式。offline inference of 隐状态。用到了过去和未来的观察，也就是Forward-Backward算法  
Fixed-lag smoothing方式，online inference，固定延迟一小段时间，也就是过去和一点点未来的观察  
Prediction方式。Infer未来的隐状态，预测一小段未来。MAP序列，称为Viterbi算法  
  
### Python代码  
04_casino_hmm_inference代码，用到了jax包，所以windows下不太好用，也可以用WSL/Google Colab来做，但是最好用linux  
识别是用的正常骰子还是有配重的骰子  
还有讲解模型训练的04_casino_hmm_learning代码  
  
# 2.15 第五课 Sampling  
网课没搞好，错过了一些开头复习内容……  
## 5.1 引入  
图像降噪，每一个像素都是原始清晰图像的隐变量？  
有了噪声图像y，希望恢复原始图像z，可以使用贝叶斯，计算后验概率p(z|y)最大化。然后方法也是最大化似然函数p(y|z)，假设每个像素都是被独立的高斯噪声影响，则$p(y|z)=\prod_j p(y_j|z_j)=\prod_j N(y_j|z_j,\sigma^2)$，就可以做MLE。  
但想做MAP的话，最大化p(z|y)=p(y|z)p(z)/p(y)有点不好做，引入了先验概率p(z)，但是怎样的先验概率才是合理的？不能随便说每个像素之间都是独立的了，相邻像素之间是有关系的，p(z)就会有点复杂。但这样就会让p(z|y)变成一个高维的分布很难处理，所以这里引入sampling的方法，想办法近似p(z|y)来简化模型。  
### 为什么要sampling？  
老师这里的引入有点迷，看不太懂这个图像降噪的问题，还是抽象一点整理下概念。就是回答为什么前面都在介绍模型，这里突然开始讲采样了？  
因为学习模式识别，要用不同的概率模型去拟合数据分布做预测，所以要学模型。但是要通过训练数据训练确定模型参数，一般需要用到贝叶斯推断的方法，而在贝叶斯推断中面对复杂的高维分布就很难做，没有办法都求出解析形式来，那换一种思路，干脆抛开对于后验概率计算的追求，直接用采样的方式来近似得到想要的结果进行决策。  
### 贝叶斯推断的局限性  
* closed form难得  
在贝叶斯推断中，后验分布正比于先验分布和似然函数，理论上随便给先验分布和似然函数的形式就可以得到后验分布的表达式。但实际上复杂高维的形式根本算不过来，要想得到后验分布的明确数学表达式（英文叫closed form），一般只有（还有其他情况也可以）似然函数和先验分布满足conjugate prior（复习下：后验和先验分布共轭时，先验分布称为似然函数的conjugate prior）或者似然函数和先验分布都是exponential family likelihoods（指数族分布，其实是一个比较宽的概念，所有概率密度可以化成一个形式挺复杂的指数式子（懒得记了）的分布都叫做指数族分布，包括了高斯、伯努利、二项、泊松、Beta、Dirichlet、Gamma等一堆常见分布）才好做。  
* 限制了我们模型的选择  
* 难处理高维分布  
### Sampling的理论基础——大数定律SLLN  
为了解决上述贝叶斯推断局限性，思路很简单，就是改用蒙特卡洛方法。我拓展查了下资料，看这个东西可以看做是贝叶斯推断的一种改进，也可以看做是抛开贝叶斯推断的基于数值采样的近似推断方法。（approximate inference）。这里有个重要的概念要理解，贝叶斯推断的核心是找到后验概率（希望有解析解，做不到退而求其次引入采样来近似），而近似推断中可以直接对需要的未知参数进行点估计。  
具体做法就是生成一些服从后验概率$p(x|\theta)$的采样值x_1,x_2...，基于采样值计算任意感兴趣的值，可以去估计x的后验概率密度$p(x|\theta)$，如果给的是f(x)也可以估计f(x)的后验概率密度$p(f(x)|\theta)$，或者最常估计的是$E[f(x)|\theta] \approx \frac{1}{n}\sum_{i=1}^nf(x_i)$……这个理论基础是大数定律。  
### 学习采样的思路  
这一章首先会介绍低维下的标准分布采样方法以及monte Carlo采样（这个定义有点迷，自己查了下，广义而言直接采样的方法+拒绝采样+重要性采样+MCMC等等都叫做Monte Carlo方法，不过概念无所谓吧）  
然后下一章介绍高维分布采样的MCMC Markov Chain Monte Carlo方法  
标准分布的采样方法是基础，复杂方法的subroutines子程序也基于这个基础方法  
什么是采样？就是获得一个随机变量的一些实际值，随着采样数的增加，抽样分布的直方图其实会和随机变量的pdf越来越接近。但不好的采样方法则可能无法近似。  
## 5.2 标准分布直接uniform采样  
如何做好的采样？那就是要均匀的搞。怎么均匀的搞？补充下cdf，就是累积分布函数F，F(x)=P(X<=x)。这里证明了个$X=F^{-1}(U)$且U属于[0,1]的均匀分布情况下，则$P(X \le x)=P(F^{-1}(U) \le x)=P(U \le F(x))=F(x)$，也就是找到一个百分之多少的点，就可以找到对应分界点的x，从而证明X是F确定的分布？所以抽样X就可以直接从F采样，再因为U和F可以对应，那只要学会在U中间标准采样，就可以转化为对F的采样，例如Python里的random.random()方法就可以[0,1]采样，当然这是伪随机的。  
### 补充cdf pseudo-inverse的问题  
就是如果F是有一段平的，那$F^{-1}$就可能无法一一对应了，会竖直跳一段，也就是U的一个区间对应X的一个值。这也不影响，这个时候就叫pseudo-inverse？  
### 例子  
#### 例1 指数分布采样  
X服从指数分布，得到F(x)表达式$F(x)=1-e^{-\lambda x}$，然后由U=F(x)方程一通变形可以反过来得到X=F^{-1}(U)的表达式，即$X=-\frac{1}{\lambda}\log (1-U)$，利用这个式子再去采样，CDF取值范围就是从0-1，所以就可以生成一个服从0-1之间均匀分布的U来采样X了。  
#### 例2 伯努利分布采样  
伯努利分布X是离散的取值，所以F(x)就是两段平着的了，所以U不再一一对应X，而是$U \in [0,1-\theta]$对应X=0，其他情况X=1，也就是所谓的pseudo-inverse  
### 变换Transformations  
如果是Y=f(x)，那可以从X的概率分布得到Y的概率分布，$p_X(y)=\sum_{k=1}^K\frac{p_X(x_k)}{|f'(x_k)|}$，其中x_k是方程f(x)=y的解，可能有多个。这里不太好证明有点技巧老师略过了，感兴趣自己去了解。  
需要知道各个点的导数也不一定能做。  
#### 变换例子  
##### 例1 线性关系  
Y=aX+b，解的$x=\frac{y-b}{a}$，而且f'(x)=a，则代入公式得到$p_Y(y)=\frac{1}{|a|}p_X(\frac{y-b}{a})$  
如果p_x是0-1的均匀分布，则根据公式$p_Y(y)=\frac{1}{|a|}$，当$\frac{y-b}{a} \in [0,1]$的情况下，其他为0。算一下结论是变成了Unif[b,a+b]  
##### 例2 平方关系  
Y=X^2。有两个解x=+-根号下y，f'(x)=2x  
TODO：但公式里的分母|f'(x)|如果为0了怎么办？  
#### 变换与采样  
这里听着听明白的，后面一想我就蒙了，为什么在采样这里讲个变换？另外研究了下，发现这里其实也可以叫做变换采样。就是对于Y的概率分布不好采样的话，可以通过X做 TODO：具体怎么做？  
## 更复杂的分布采样  
如果可以拆分为多个简单分布的和，那就可以利用已经学习的标准分布uniform采样+变换的方式分开采样，然后加起来  
给了个例子是Gamma分布可以拆分为指数分布的和  
但是这样的思路适用范围依然很优先。例如Gamma分布里的a非整数就做不了，稍稍复杂一点的分布就做不了，所以继续介绍高级方法。  
## 5.3 Rejection采样  
如果要采样p(z)，知道$p(z)=\frac{1}{M}\hat{p}(z)$，但常数M不知道（为什么要支持这个常数M我一开始很蒙蔽，我想着既然已知$\hat{p}(z)$，根据概率密度积分为1的特性，不就是积个分就求出来M了，后面其实看到拒绝采样最后拿去分析后验概率就知道了，这个M对应的是分母——样本的先验概率p(D)，我们只关心似然乘先验就能做MLE、MAP了，简单情况能去积分似然乘先验，但是复杂情况分布可能很复杂犯不上啊，之所以要做采样就是不想求解析式）。  
可以找个q(z)容易采样的，再找个常数k使得$kq(z)\ge \hat{p}(z)$对于所有z都成立（也就是kq(z)的图像可以把$\hat{p}(z)$包起来），数学描述是$supp p \subset supp q$，这里的supp p的意思就是包含的区域吧？没太看懂  
### Rejection采样流程  
* 1从比较好采样的提议分布q(z)中采样z  
* 2从Unif[0,kq(z)]中采样u（看图会比较直接，就是x轴上找到z点，然后在高度最高到kq(z)的线段上均与分布找一点）  
* 3如果$u\le\hat{p}(z)$即在原分布内则接受u，反之则是虽然落在了kq(z)内，但超过了原分布，拒绝。  
  
可以分析下横轴上一个点的采样u接受概率是$\frac{\hat{p}(z)}{kq(z)}$，整体而言接受z的概率是等于$\frac{M}{k}$，证明这个就是把横轴所有z对应的竖轴接收概率都积分一遍……$P(z \text{ accepted})=\int \frac{\hat{p}(z)}{kq(z)}q(z)\mathrm{d}z=\frac{M}{k}$  
总而言之k也不能搞太大，越大接受率越低，能正好包住\hat{p}(z)是最好的。另外x轴上也是需要更大的，这里也是选大了可以做，但是接受率会减少。  
### Rejection采样可行性证明  
这个我还挺懵逼的不知道怎么证，看起来要证明一个采样方法是可行的，也就是证明基于采样结果的CDF等于发生的概率$P(z\le z_0|z \text{accepted})$  
$P(z\le z_0| z accepted)=\frac{P(z\le z_0,z accepted)}{P(z accepted)}=xxx=\frac{1}{M}\int_{z\le z_0} \hat{p}(z)\mathrm{d}z$  
中间推导就是纵轴u从0-$\hat{p}(z)$内层积分，外层再横轴从负无穷积分到$z_0$……最终的结果就是p(z)的cdf，所以用接受的u作为采样是可行的  
### Rejection采样处理Gamma分布  
如果Gamma分布的参数a是整数，可以拆分为多个指数分布去组合，但是非整数的情况下拆分就做不了了，但是可以做Rejection。  
结论是最好的选择是q(z)用Cauchy分布，其中参数c和γ都按指定的公式（式子有点复杂懒得记了）。然后可以证明能取到最小的k值按指定式子。  
### Rejection采样例程  
老师讲了rejection sampling的python代码，应该是用一个大正态分布q(z)去rejection两个小的正态分布组合的p(z)  
### Rejection采样用于贝叶斯推理  
从Rejection采样的角度来看看贝叶斯推断求后验概率的问题  
$p(\theta|D)=\frac{p(D|\theta)p(\theta)}{p(D)}$  
如果要解析的求出来，那分母p(D)是个复杂的高维积分？不好算，$\hat{p}(\theta)$也需要和后验概率共轭分布。  
反正分母是个不好分析的概率分布p(D)，但是是常数，正好当做M给忽略掉~那$\hat{p}(\theta)=p(D|\theta)p(\theta)$似然乘先验。我们再设提议分布等于先验概率分布$q(\theta)=p(\theta)$，那k值等于$\frac{\hat{p}(\theta)}{q(\theta)}$就是似然函数，不过要对于最大的似然函数也成立，也就是MLE时的似然函数取值。  
这个结论就可以让我们用拒绝采样来搞贝叶斯估计的后验概率了。  
  
## Importance Sampling  
Importance Sampling某种意义上说其实就是加权版本的拒绝采样，从而实现了不用拒绝直接乘权重就可以。  
老师介绍Importance Sampling一般不是用来估计p(z)的（当然也可以做），用它来直接估计f(z)的期望很方便（例如EM算法的E步骤中）  
由大数定律可以知道可以用p(z)的采样值z_1,z_2,...,z_n去求平均值来近似期望  
更好的思路是从|f(z)|p(z)中采样z，会比从p(z)中采样更有效率。因为例如$f(z)=1 (z \in E)$，E是个小概率事件，那就很难采样遇到E事件。我们希望引入importance weights的形式  
### Importance Sampling步骤  
假设p(z)不好采样但是q(z)好采样，而且supp p $\subset$ supp q，可以证明$E_p[f(z)]=\int f(z)p(z)\mathrm{d}z=\int f(z)\frac{p(z)}{q(z)}q(z)\mathrm{d}z=\int f(z)w(z)q(z)\mathrm{d}z=\frac{1}{n}\sum_{i=1}^nw(z_i)f(z_i)$，其中$\frac{p$  
现在从q(z)中采样，但是现在不用Rejection了，不需要考虑q(z)比p(z)一直大，但是需要保证f是在q(z)的定义域内，即supp f $\subset$ supp q（这里也写的supp，前面拒绝采样的包络也写的supp，感觉应该区分下才对）。  
重要性采样的过程是误差不断减少的过程，最终convergence的表现取决于q(z)和|f(z)|p(z)匹配的程度。（TODO：为什么？乱了……）  
#### importance Sampling例子——长尾采样求长尾概率  
考虑Tail Sampling，P(X>a)是长尾，概率很低  
$P(x\gt a) \approx \frac{1}{n}\sum_{i-1}^n w(z_i)$  
使用importance Sampling就可以做，没跟上，给了python代码  
#### p(x)未归一化问题  
但是$w(z)=\frac{p(z)}{q(z)}$，其中p(z)如果是没有归一化的$\hat{p}(z)=Mp(z)$，和拒绝采样中介绍的一样，那看起来会有问题，需要用点技巧，在不知道M的情况下来做归一化。  
推导一通得到$w_n(z_i)=\frac{w(z_i)}{\sum_j w(z_j)}$，这里是normalize后的权重，就不需要知道p(z)了，只要知道$\hat{p}(z_i)$就行了  
### Sampling Importance Resampling SIR  
算是做重要性采样的另一种思路，重要性采样的核心思想是考虑了不同样本的权重不同，如果一个样本在原分布p(z)中出现概率低，在采样用的提议分布q(z)中出现概率低，那就降低其权重，实现了能采样出来一些小概率点还控制有限影响的效果。  
换一种思路，如果就是想从不做权重的p(z)中采样呢？理论上就是按照p(z)采样，出现概率大的点多采样才更准确嘛，这怎么做？那就先去重要性采样，然后去替换，权重值可以等同于保留不替换的概率，那么权重低点就容易被替换掉，权重高的点就更容易保留，最后只要采样点n足够多，那保留的样本点分布就会接近p(z)。  
这种方式可以对比下拒绝采样（老师没讲自己随便查一点不一定准确），同样是按照p(z)去采样的，但是不存在拒绝的问题，是一上来就有重要性采样得到的可能误差较大的结果然后慢慢去迭代更新，应该效率更高，算是兼顾了拒绝采样和重要性采样的优势，不过计算量会大一些。  
步骤是：  
1.从q(z)中采样z_i  
2.计算权重w_n(z_i)  
3.重新采样替换z_i，根据权重w_n（TODO：具体细节这里没有深入）  
#### SIR原理证明  
也是推导$P(z\le a)$的形式，老师证了半天，脑子已经跟不上了。  
#### SIR做贝叶斯推断  
反正也是啥计算权重，然后根据权重去重新采样θ，就可以不需要知道p(θ)  
可以推导一下，待采样的是似然函数乘先验概率，也就是M倍的后验概率，$\hat{p}(\theta)=p(D|\theta)p(\theta)$。提出提议分布和先验概率一致，$q(\theta)=p(\theta)$。算一下自带归一化的权重值$w_n(z_i)=\frac{\hat{p}(\theta_i)/q(\theta_i)}{\sum_j\hat{p}(\theta_j)/q(\theta_j)}=\frac{p(D|\theta_i)}{\sum_jp(D|\theta_j)}$，很神奇先验概率p(θ)就被消掉了  
### 再补充引入下EM算法的采样  
想采样p(z|x,θ)，这个不好做，高维分布了，所以下节课MCMC再处理。  
  
# 2.22 第六课 MCMC  
回答上节课的一些问题。  
* 1.如果做rejection采样不知道$\hat{p}(z)$怎么做？  
$M=\int_{-\infty}^{\infty}p~(z)\mathrm{d}z$  
实际中不好做，直接用un-normalized $\hat{p}(z)$  
* 2.为什么要standard sampling，既然已经知道了pdf  
这个是做其他sampling的基础，rejection、importance、MCMC都用  
* 3.importance采样中说不要求q(z)不p(z)大，但是又要求supp p是supp q的子集，怎么做到的？  
这里supp应该只是区间是子集就行，不要求面积包括住。因为q(z)可以乘以系数k去扩大面积。  
* 4.importance采样中说汇聚效果取决于q(z)和|f(z)|q(z)的匹配程度，为什么？  
$E_p[f(x)]=\int_{-\infty}^{\infty}f(x)p(x)\mathrm{d}x$，|f(x)|p(x)越大，对于积分的贡献越大。可以证明最优的取得最小采样误差的q(z)正比与|f(z)|q(z)  
* 5.为什么q(z)采样z_1,z_2,...的support是$(a,\infty)$，和importance权重一起，就有$P(X \lt a) = \frac{1}{n}\sum_{i=1}^nw(z_i)$  
因为这个等于$E[f(x)]=\frac{1}{n}\sum_{i=1}^nw(z_i)f(z_i)$，f(z_i)在大于a的时候都是1  
* 6.SIR中，为什么重新采样数据只有m个采样？有n个q(z)的采样，重新采样为什么不是n个  
这里老师更正了一个式子？见scripple。  
m个数据是从基于不同不同权重的z_n的multinomial分布中取得，是多少都可以？反正按照权重  
  
## 引入  
图像分析，观察到的是有噪声的y，实际图像是z，想知道z是什么样，用MAP $p(z|y)=\frac{p(y|z)p(z)}{p(y)}$，所以就需要知道p(z)的先验概率。  
但是这是一个高维分布（TODO：啥？），不好分析，直接就把p(z|y)也整成高维分布求不出解析形式了。所以希望不去算解析式了，直接从p(z|y)中采样。  
上节课学过的方法只适用于低维的pdf，高维不太行，要用MCMC这个可以从高维分布中采样的方法。  
这是20世纪10个最重要的算法之一。最早是从物理化学领域来的，后面统计学开始分析，研究了前身Gibbs采样，后面1990年流行起来，也是有算力进步的原因。  
注意不是啥马尔科夫链都可以拿来用，特殊的马尔科夫链所具有的性质才可以帮助做随机采样。MCMC基于马尔科夫链的stationary distribution平稳分布状态。  
### 马尔科夫链可能具有的特殊性质  
简单来说，可能有很多形式的马尔科夫链，但我们想要的是最普遍，具有较好的数学性质马尔科夫链，这就要根据特殊性质挑选出来。  
* irreducibility 不可约性  
数学角度描述是状态空间仅有一个连通类，或者说从任意一个状态到任意另一个状态都有概率在有限步内实现。说白了就是没有那种单向接入的状态走出去永远回不来的。  
* recurrence 常返性（有点奇怪的翻译，感觉叫循环性也行）  
常返态分析的是从一个状态i出发，未来有多大概率会回到这个状态。  
首先返回概率为1叫做常返态，返回概率小于1叫做非常返态（对应的情况就是自己这边有个循环的圈，但是可能经过某一个单向路径到另一个圈就永远回不来了，所以能一直本地转就能返，跑丢了就不能返）。  
然后常返态又分为positive正常返态和null零常返态，MCMC关心正常返态，即有限时间内就能返回，而零常返态是理论上能返回但时间趋于无穷（这个不好理解，但实际上只对应无限长的马尔科夫链，也就是一直往下走的概率大于往回返的概率）。常返性和不可约性还挺有关联，简单而言有限节点的不可约马尔科夫链一定是正常返的，这个想想也知道就是个有限的连通图互相转。  
* 周期性  
这个只针对于正常返态的马尔科夫链，  
根据返回时间是否是周期性的，例如从i出发返回i可能的步数是3、6、9、12……就可以分成周期态和非周期态。  
TODO：我没能从几何上理解周期性是什么。感觉像是对应有固定路径的环。  
  
综上，直观感觉我们不希望要一些稀奇古怪的马尔科夫链，如果一个马尔科夫链集齐了正常返性+非周期性，就是ergodic遍历性的马尔科夫链，是普遍意义上适合研究应用的马尔科夫链~  
  
## stationarity 平稳性——做MCMC理论基础  
### stationary distribution 平稳分布  
考虑一个homogeneous的马尔科夫链，转移概率T(x,y)  
π就是stationary分布，如果对于所有的y状态满足$\sum_x \pi(x)T(x,y)=\pi(y)$  
也叫做invariant分布，分布不随着MC处理改变。  
如果存在这样的平稳分布状态，那一方面就可以深入分析稳态的结果，另一方面马尔科夫链也可以拿来作为一个特殊的概率分布用，听起来不错。但是并不是所有的马尔科夫链都会有平稳状态。  
假设有M个状态，写出全部的T矩阵就是MxM的。π满足$\pi T = \pi$，这就是特征值和特征向量。可能是无解的，要分析下是否存在。  
### Stationarity的Condition  
直接给结论，存在stationary分布的condition：  
1.马尔科夫链是irreducible的，任何点都可以到任何点，不会有回不去的点  
2.马尔科夫链是positive recurrent的，从任何状态i出发，都是有限时间内可以回到i状态  
这里做个题练习下，给定T求π就是特征值特征向量问题。  
有限状态的irreducible马尔科夫链一定是positive recurrent的，也就是一定有平稳状态  
#### Asymptotic Steady State 渐进稳态  
这个算是更进一步的性质，即不光稳态存在，而且从任意初始状态不断迭代后最终都会达到稳态。  
渐进稳态的条件是既满足stationary分布存在的条件，还需要非周期aperiodic。前面也说了这种马尔科夫链叫做ergodic遍历的，有很好的性质：在足够长的burn-in周期m之后，k>m的分布都近似stationary distribution。这个性质很关键。  
### Reversible MC 更特殊情况——可逆马尔科夫链  
可逆马尔科夫链是马尔科夫链有稳态的充分但非必要条件，性质更好~  
$\pi(x)T(x,y)=\pi(y)T(y,x)$就叫做Reversible的马尔科夫链，也就是两点之间互相转移的概率是一样大的。  
所以如果能构造Reversible的马尔科夫链，再做成非周期的（有渐进稳态了），就有了非常好的性质：构造马尔科夫链采样，当采样点数n足够大的时候（进入稳态），就可以通过采样z来近似目标分布π。  
为什么？这个下面看具体的MH算法才能理解。  
## Metropolis-Hastings Algorithm  
采样π(x)很难，也是如果能计算简单的un-normalized版本的π(x)也就是$\tilde{\pi}(x)$，在选择转移概率q(x,y)，也是X上的irreducible和aperiodic的，就容易采样，叫做proposal分布  
介绍了一下步骤，啥从q(x,y)中采样y，以一个A(x,y)的概率接受y？（这个A(x,y)怎么来的很重要，老师另外补充了一页证明），如果接受y，$Z_m=y$，不接受$Z_m=x=Z_{m-1}$TODO：没跟上  
Z也是个马尔科夫链，  
### MH算法证明  
证明这是个Reversible的马尔科夫链？  
### Proposal Distribution——q(x,y)怎么选？  
Proposal分布的选择可能影响性能，这里有一个很聪明的思路~  
#### 对称的q(x,y)和q(y,x)  
因为看的是从x转移到y，我从起点x的角度看，出发的移动的距离就是y-x，我们就可以考虑y-x的分布，q(x,y)=q(y-x)。  
关于y-x一些典型的选择有以x为中心的高斯分布或者以x为中心的均匀分布。这些对称的选择有一个极好的性质，q(x,y)=q(y,x)，所以A(x,y)的形式消去了q(y,x)/q(x,y)，变成了非常简单的$\min(1,\frac{\hat{\pi}(y)}{\hat{\pi}(x)})$，这就是Metropolis算法。  
#### 忽略马尔科夫链直接q(y)  
还有更简单的选择，就是q(x,y)=q(y)，叫independent chain，反正一通推这个性质会使得A(x,y)更加简化。  
#### 更一般的简化，拆分易采样h(x)和有界$\psi$(x)  
分析下一般情况能不能简化，如果$\hat{\pi}(x)\propto \psi(x)h(x)$，也就是可以拆分为一个有界函数$\psi$(x)和一个易采样的函数h(x)的乘积，那么就可以取q(x,y)=h(y)，这样一化简就把h(x)全都给消掉了，也可以简化A(x,y)形式为$\min(1,\frac{\hat{\psi}(y)}{\hat{\psi}(x)})$  
#### q(x,y)例子：对称的高斯分布&讨论σ影响  
给了个例子，π(x)和一个略复杂的一维分布相关，然后选择q(x,y)为高斯分布简化形式，还是挺好解决的。补充下：具体操作决定A(x,y)的转移概率要不要跳转的时候，直接做个0-1均匀分布的采样点也就是拿一个随机数，然后拿去和A(x,y)作为的阈值比较。  
这里可以发现根据高斯分布的σ不同，会影响接受率和最后采样的结果，合适的值才行，不然马尔科夫链很难顺利的随机游走，σ太小可能就固定一块游走，σ太大可能极难走到合适的接受率很低  
一般而言随机游走的MH接受率在0.2-0.5，不过直接用independent MH接受率就是1了。  
### Burn-In  
burn-in阶段是从初值走到接近于stationary分布的过程。不过挺难确认什么时候结束的。  
一般写代码是跑n步，然后抛弃，这叫做burn-in period，一般是前1000-5000步都抛弃。  
### Thinning  
这个操作是打破通过马尔科夫链采样造成的前后采样点之间的dependence，可以每走d步采一次样。这样当σ取得太大了导致长时间卡在一个位置的时候，用这种降采样的策略就可以有更好的效果  
那这个d怎么选呢？可以深入分析下MH算法中的采样点关联性，一通推导后可以得到lag t之后的auto-correlation的值的表达式，式子有点复杂不记了。  
可以参考python代码06_mcmc_gmm_demo.ipynb，模拟计算了不同d对应的autocorrelation，看起来d=40左右不错基本在0附近了。  
  
## Gibbs Sampling  
是MH算法的一个特殊情况  
想从p(z1,...,z_d)的多维分布中采样，如果高维很困难  
我们想直接从full conditionals的$p(z_i|z_{-i})$中采样，能够轻松解决高维的问题  
这个公式表达起来比较难懂，但是几何上会很清楚，看最简单的二维情况，如果正常做每一步就是从一个点A转移到一个点B，x和y坐标都会变。现在改成移动下一个点的时候依次只动一维，这里就是先只走x（y轴方向平移），再只走y（x轴方向平移），然后依次循环。可以理解为MH中的一步变为了n维的n步。  
### Gibbs采样流程  
也是初始化一个初始向量，  
然后每一轮次，逐个从$p(\cdot|z1^{(k-1)},...,z_d^{(k-1)}\text{not include self})$采样$z_i^{(k)}$，不含x_j的d维向量可以记为$x_{-j}$，这样多维采样就变成了多个一维采样。  
### Gibbs采样证明  
可以证明这实际上是一个A(x,y)=xxx的一个MH。  
明明是从一个条件概率中抽样，为啥就可以近似？  
### Gibbs采样细节  
#### 消除采样点相关性  
两种做法，一是生成r个长度m的Gibbs序列（r和m是啥？），二是我们比较熟悉的，生成长的序列抛弃burn-in阶段的值，然后r次采样才取一个去除关联性  
#### 如何获得条件分布？  
这里具体的做法是忽略所有不依赖z_j项的值，直接说不太懂，还是要看例子  
看个例子，给了个具体的p(x,y,n)，然后分别给出了p(x|y,n) p(y|x,n) p(n|x,y)，要p(x|y,n)就是很简单从p(x,y,n)中抹除不含x的一些值（y和n相关的还保留，有点晕）  
### Gibbs采样in Ising Model  
再来个例子Ising Model，还是开头说的图像去噪的问题，没跟上。为了后验概率需要计算Ising先验分布，然后采样p(z|y)，使用gibbs采样，先计算conditional分布。这里也是给了对应的python代码可以学习下。代码里还有个meanfield方法，会不太一样。  
还推荐一个视频是MCMC Training of Bayesian Neural Networks，想不明白是怎么做的。  
  
## 总结  
还有个Sampling整个的python代码，可以从代码出发去认识各个采样方法具体的计算。  
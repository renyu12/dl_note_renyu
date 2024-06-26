# 参考资料  
笔记内容主要来自于SFU Jie Liang老师ENSC 424课件、吴恩达DeepLearning.ai的机器学习&CNN课程以及一些技术博客内容、B站up主分享，混合整理了下。  
  
另外推荐（自己没看仅记录）  
斯坦福、多大、普利斯顿、CMU的课程  
Neural Networks and Deep Learning的书、Dive into Deep Learning的书  
  
# 学习思路  
回顾起来深度学习入门内容还是涉及的知识点有一些杂乱，也没有什么特别平滑高效的学习方法，还是得学习一段后才能串联起来。Andrew老师课程从传统机器学习引入先介绍了一些基础知识再引入会系统一点，流程比较长。Jie老师课程从神经网络的发展流程开始引入，需要的基础知识再针对补充会效率高一些。  
## Andrew老师课程  
（前置课程从传统机器学习引入，对回归、分类问题；代价函数；梯度下降有基本的认识）机器学习的分类 有监督/无监督 -> 有监督分类、回归 -> 最简单的线性回归问题 -> 引入优化问题代价函数 -> 引入梯度下降解决线性回归问题 -> 继续升级到多输入的多元线性回归，思考多项式的非线性回归 -> 最简单的分类问题 -> 引入sigmoid函数和交叉熵损失函数解决从回归到分类时离散输出导致的新问题（逻辑回归） -> 逻辑回归一样用梯度下降方法 -> 补充过拟合问题（数据量）（特征选择）（正则化）  
  
（神经网络部分）神经网络发展历程 -> 模拟生物的神经产生了神经网络的设计 -> 神经网络快速发展原因（大数据+深度神经网络=更高上限）-> 神经网络结构与正向传播运算流程 -> 神经网络正向传播代码实现 -> 补充矩阵乘法+GPU获得的高效率 -> TensorFlow构造神经网络 -> 分析激活函数 -> 多分类与Softmax -> 梯度下降优化  
  
（CNN部分）计算机视觉应用 -> 二维卷积的过滤器作用 -> 填充&步长 -> 三维卷积 -> 卷积层 -> 简单的卷积神经网络 -> 池化 -> 回顾卷积意义 -> 经典网络（LeNet AlexNet VGG ResNet GoogLeNet……）  
  
## Jie老师课件  
（从简单感知机到复杂的MLP）神经网络发展历程 -> 最简单的神经元，感知机模型 -> 感知机实现或门问题 -> 引入感知机缺陷与数学意义，线性分割 -> 异或门问题 -> 引入多层感知机 -> 讨论训练方法，给出了激活函数和BP算法（这里会跨度较大） -> 分析神经网络训练的优化（批量、小批量、随机）（学习率调整）（冲量梯度下降）-> 分析神经网络过拟合问题（权重衰减（正则化））（训练数据量）（控制隐含层） -> 补充划分训练、测试、验证数据集方法 -> 梯度消失、爆炸问题  
  
（CNN部分）卷积从一维到三维 -> 池化 -> 经典网络  
  
  
# 正式内容  
## 概述  
### 人工智能发展简史  
1956年就提出了，研究很火热发展很快，第一代神经网络感知机也是此时提出，但是1974-1980发现陷入瓶颈逐步停滞。  
80年代提出了第二代神经网络多层感知机（MLP即全连接神经网络）以及BP反向传播算法，能解决非线性问题，后面因为梯度消失层数有限难以训练的问题，发展又陷入停滞。  
直到06年提出深度学习，使用无监督的学习方法逐层训练，然后再BP算法反向调优，层数增加效果更好，让大家看到了希望。随着算力还有大数据的发展，2012年加深卷积层数的AlexNet爆杀ImageNet，又进入火热发展的第三阶段。  
  
### 深度学习成功的原因  
  
+ 算法的突破  
+  GPU硬件发展  
+ 可供训练的大数据集  
深度学习相比传统机器学习，个人认为重大的突破是实现了：用增加数据量换取减少人工特征提取。这样就实现了对于很多复杂问题的处理，因为随着问题越来越复杂，需要的特征也会逐渐复杂，人工特征提取很需要创造力，就和搞科研一样是需要根据现有知识分析，产生灵感再实验验证的，有点像随机地搜索，但是增加数据量可以大力出奇迹，把所有可能的特征都拿出来用数据验证，慢慢筛选出可用的屏蔽无效的，类似地毯式搜索。  
+ 网络、社交媒体（提供最新论文、学习资源、交流平台）  
  
### 机器学习分类  
深度学习是机器学习中的特别一支  
+ 监督学习（回归和分类）  
训练数据有标签（目标输出），输出集合离散有限是分类问题，无限是回归问题。本质上是让模型去模拟一个函数，模拟x->y的映射关系  
+ 无监督学习  
训练数据无标签，所以也做不了输出的预测，一般目标是发现输入数据中的结构性structure  
+ 强化学习  
没有训练数据，但是动作会得到反馈  
  
## 感知机到神经网络  
### 引入——思考生物的神经是怎么工作的？  
神经元有多个突触输入，有的可能刺激，有的可能抑制，最终决定输出的结果，继续影响后面的神经元  
所以仿照这个，可以搞一个数学模型来模拟一个神经元。例如函数也可以实现多输入对单输出，但是不好实现输出分类，所以提出了感知机perceptron（相当于一个线性函数后面接一个激活函数），就是神经网络的基本雏形  
  
### 感知机数学原理  
多个输入$x_i$，每个权重$w_i$不同，累加得到结果s。感知机有阈值T，如果s>T就输出y=1，反之输出y=0  
    多个感知机可以组成一层，获取多个输出$y_j$  
    有对于输出$y_j$的训练用例结果$t_j$，就可以学习调整权重$w_i$。调整的原理很简单，就是比较yj和tj，如果不对了，例如$t_j=1$，$y_j=0$没到达阈值，那么就对于$x_i>0$的增加$w_{j,i}$的权重，$x_i<0$的就应该降低$w_{j,i}$的权重。（这样是一个用例就把所有权重都调整了，不过测试用例多的情况下应该是合理的）  
    具体调整$w_{j,i}$多少，应该是根据当前用例$t_j-y_j$的差值，还有xi的大小来决定，简单的式子就是$w_{j,i} = w_{j,i} + η(t_j-y_j)x_i$。η就叫做学习率，控制调整权重的速度。  
    这里就引入了一个问题，就是$x_i=0$的时候就不好整了，数学上也很好解决，直接加上一个偏置bias，例如给每个感知机j固定输入一个$x_{j0} = 1$或者-1，那输出$y_j$就需要加上$w_{j0}\cdot x_{j0}$，再简单一点不把$w_{j0}$放在一起调整，就认为是加上一个$b_j$就可以。  
  
### 感知机的训练  
    总结一下，就是先初始化wj,i和bj，可以设置成随机数  
    然后开始训练，训练可以有很多轮。  
    针对每一组训练用例，输入x计算出y，如果全部数据yj都和tj一样了就停止。  
    否则就根据当前的训练用例，调节wj,i，重复训练。  
  
### 感知机的数学意义 —— 线性分割  
    考虑一下用感知机实现或门的问题来推导感知机的数学意义，即两个输入x1 x2（最简单的二维情况），任意一个为1则输出y=1。  
    （关于为什么感知机是线性分割的这里没有讨论，我个人理解还是从数学公式看出来的，补充下）从感知机的数学模型上可以知道，设前面线性函数部分的输出s=0，那其实对应的是一个线性的平面（直线、平面、超平面）。  
    简单一点处理假设阈值T=0，那考虑在决策边界上的两组输入x1(x11,x12)和x2(x21,x22)，就会有wx1 + b = 0 和 wx2 + b = 0，推出w(x1-x2)=0。这里x1、x2两个点的连线其实就是决策的分隔线，w向量应该是与决策分隔线正交的。  
    所以感知机有线性分隔的能力。与门一共4个可能的输入(0,0)(0,1)(1,0)(1,1)，直接在中间划线分隔就成了。  
    但是单层感知机的缺陷也就在这里，对于非线性的问题无法处理，例如异或门问题。  
  
### 多层感知机（MLP）解决非线性分隔问题  
    思路也很简单，考虑用多个感知机来解决，并且组成多层，每一层的结果作为下一层的输入继续计算。这里激活函数是很重要的，后面讨论激活函数部分会详细分析，没有激活函数感知机做的仅仅是输入值按权重累加，无论做多少层都还是一样的效果，仍然是线性分割  
    结构是：输入层-多个隐含层-输出层  
    还是以异或门为例，如果是3个神经元，输入层2个，然后输出层再1个，这样的网络是可以实现异或门的  
    （Andrew课程补充下抽象理解）输入层是原始的各项特征，输出层是最后的结果，隐含层可以理解为一些高级抽象的特征。例如特征有服装价格、运费、品牌知名度，材质等多项，需要预测服装的销量，那价格和运费就可以组合成消费者是否买的起，价格、知名度、材质可以组合成是否划算……通过调整隐含层输入的权重即可，用这样的更高阶的特征来预测最后的结果。  
  
  
## 神经网络的训练  
### 如何训练神经网络？ —— 最优化问题  
    之前处理的或门问题很简单可以直接线性分割，所有的点都不会错，但是很多实际问题下感知机是无法做到完美分类/拟合的。那直接的思路就是尽可能做到最好，例如让分类错误的点最少，但是这个是个NP Hard问题无法解决，需要更简单可行的方式。这里的解决方法是把训练神经网络看成是一个优化问题，找到神经网络模型输出和实际问题答案对应模型之间差距的度量方式，然后用优化问题的数学方法去降低这个差距（深入学习可以多了解最优化的数学知识，有很多优化算法值得学习）。  
    所以第一步是想办法去量化神经网络模型和实际问题答案对应模型之间的差距，我们把这个量化的结果称为损失函数（loss function，或者叫做误差error函数，一般用L表示，另外这里有个概念辨析的问题，当然不需要辨析的很清楚。还定义了代价cost函数，一般用J表示。有的地方说loss和cost function一个东西，但是似乎Andrew课程中区别了，loss是针对一个样本的，而cost是针对整个数据集的，也就是loss取平均。优化问题中也经常说目标objective函数，比loss、cost更通用一些，一个东西）。损失函数的得出有一点麻烦，不过理解下还是很有意义。从简单的线性回归情况开始考虑，为了让所有预测的y和实际的t一致，这里很直接的思路就是比较输出的每一个yj和tj，计算差值然后累加，为了避免正负差值抵消做个平方，所以可以定义损失函数 $E = \frac{1}{2} \sum(t_j-y_j)^2$。也就是用的非常多的均方误差MSE（这也是常说的最小二乘法的做法），针对不同问题，可能需要用不同形式的损失函数来量化神经网络模型的误差，例如回归问题可以均方误差，但是分类问题中均方误差将不再适用，后面再详细讨论（可以从概率的角度用最大似然估计或者交叉熵的思路推出来）。  
    然后就是一个非常难搞的问题，如果训练数据有误差，按什么样的规则去调整整个网络中各感知机的权重？看做一个最优化问题，还是凸优化，最简单高效的方法就是一阶优化用的梯度下降法，重点学习。 简单而言就是从任意点出发通过求导找到梯度，即找最“陡峭”的方向前进从而到达最小值点。这里有个很直觉的理解方式，就是我们是在分析各个参数对于最终误差的贡献，贡献越大说明这个参数影响越大就需要调整更多，那求导得到在这个参数维度上误差的变化率，变得越快，我们就认为这个参数的贡献越大。  
    注意不是说梯度下降法是唯一的方法，还有和很多方法，例如二阶的牛顿法一类的，甚至特征维度不多的线性回归问题可以用正规方程法直接求解导数为0的点，但是从复杂度而言梯度下降法最为通用可行，后面也有了很多变形，在后面章节梯度下降优化部分详细讨论。  
    归纳一下，训练的过程就是根据训练数据重复（计算输出-计算误差-调整优化降低误差）三个步骤。一般难就难在第三步这里。  
  
### 梯度下降  
#### 梯度下降具体操作——从最简单的case开始  
    为了方便理解梯度下降的原理，可以考虑最简单的一种情况，输入为单个变量x，输出为y，y=wx+b，一条直线线性回归。那这里就是两个参数w和b需要调整，画出损失函数关于wb两个变量的3维图像，应该是类似一个碗一样，有一个最低点，就是最佳的w和b的取值位置。随便设置一个初始位置，求梯度就可以得到下降最快的方向，沿着这个方向就能找到最低点。当然不是凸函数的话也可能陷入局部最低点。  
    数学上的实现就是每次更新$w = w - α\frac{\partial E}{\partial w}$，其中α是学习率，b也是一样更新，只是对b求偏导，重复做直到收敛不变。  
    学习率的设定也是一门学问，很容易想到学习率低需要迭代很多次，训练太慢，学习率高搞不好直接跑过了最低点，导致无法收敛，甚至反过来发散。  
  
#### 梯度下降具体操作——升级到多维特征的梯度下降  
    形式也很简单，无非是$y = w_1x_1 + w_2x_2 + … + w_nx_n + b$。为了方便表示和计算，引入向量的乘法，w和x都是向量就非常简洁了，还是$y = \vec{w}\vec{x} + b$。numpy等库在计算向量运算的时候也做了优化可以大大提升计算效率，并且可以使用到GPU加速 并行计算等技术。  
    损失函数向量化表示也是比较简洁的形式 $J(\vec{w}, b)$。做梯度下降的时候也是一样的做法，无非是多了很多w1 w2 w3要分开求偏导更新，$w_j = w_j - α\frac{\partial J}{\partial w_j}$。  
    多个特征输入就涉及到取值范围各不相同的问题，例如x1取值范围很大，x2取值范围很小，设置和训练调整参数的时候就有一些技巧。理论上取值范围大的特征对应的权重应该更小，反之更大。通用的做法是做特征缩放，一般是归一化统一范围，例如做平均值归一化（特征x减去平均值除以取值范围长度），Z-score归一化（特征x减去均值除以标准差）。没有绝对要求各个特征的取值范围都完全统一，大致是一个数量级就行，比较灵活。  
    回到学习率的设置，可以参考横轴为迭代次数，纵轴为损失函数的学习曲线，正常情况下应该随着迭代次数增加，误差越来越小，最终趋平，如果增加了认为算法有误或者学习率设置不当。完全迭代到J不变收敛了可能迭代过多，一般设置一个ε，当前J比上一次J下降的小于ε了认为基本收敛了可以停止。实际设置的时候可以少量迭代测试不同的学习率观察学习曲线，找到一个使学习曲线平稳快速下降的就好。  
    对于特征选择深入一点分析有很多门道和经验，总结归纳成了特征工程这门学问。简单举个例子，特征工程中特征的选取就很重要，在预测房子价格时，特征选房子的长和宽两个就不如选面积一个。  
    最后考虑升级到非线性的回归问题，解决方法也很简单，输入用多项式做回归就可以，平方根 平方 立方……都是曲线。  
    推荐机器学习库scikit-learn提供很多算法。  
  
#### 分类问题怎么做梯度下降  
    讨论最简单的二分类问题，需要输出y=0/1。输出y是离散的就引入一个连续输出转化离散输出的问题，这个时候无法用线性回归的方法来解决，是可以强行把y<0.5当做0，y>=0.5当做1，但是考虑训练样本中有一些x取值极端的点，因为y取值范围很固定，就会很容易把拟合出的线性分割线/面给带跑偏，导致分割的边界错误。这里就引入数学性质非常棒的sigmoid函数来协助解决这样的问题，因为这个函数只在x=0附近很陡峭，可以作为分割线，偏离多的x取值获得的y都是接近1/0的，就非常合适，不会受到极端x取值的干扰。sigmoid函数输出值在0-1之间，直接可以视为y=1的概率（这个不知道有没有严谨论证？）。把sigmoid记录为g(z)，最终y的形式就是y=g(wx + b)，也就是复合一个sigmoid。这个操作就很巧妙，可以思考下它的几何意义，z = wx + b = 0其实就成为了输出y是0还是1的分界线，也就是所谓的决策边界，正好z>0 y>0.5，z<0 y<0.5。对于复杂需要曲线来分界的问题，之前也分析过，那就需要z是多项式的形式了。  
    接下来就是损失函数的问题。也是不能直接用均方误差了，因为输出离散，这样会使得损失函数不再是凸函数，，跑不了梯度下降……（判断凹凸性的证明求二阶导即可，这里计算稍有点复杂，结论是二阶导不是恒大于0的所以是非凸函数。另外还可以感性认识，一是分类问题的y取值非大小概念，使用代表欧式距离的MSE没有意义；二是0和1相差不大，对于分类错误点惩罚不大；三是sigmoid会有y接近0和1时梯度消失的问题，如果出现sigmoid的导数$y=(1-y)*y$就会更严重，使用MSE在求偏导算梯度的时候就会出现sigmoid的导数形式）直接给结论是可以用log形式的另一种方式计算误差就是凸函数（可以求二阶导确认恒大于0）。形式是$J(w,b)=\frac{1}{m}\sum_{i=1}^{m}(-y\log f(x)-(1-y)\log(1-f(x))$。仔细分析下，就是y=0的时候，x离1越近，误差越趋近于无穷大，反之y=1的时候，x离0越近，误差越趋近于无穷大，是对数形式增加极大增加了惩罚。补充展开下这个log形式的损失函数是怎么来的，一种思路是最大似然估计，也就是概率统计中根据当前的样本（训练数据）分布去反推整体模型的概率分布的做法，说白了就是认为抽出来的样本是已经发生的事件，那就认为抽出来当前样本是可能性最大的才会发生挺合理的，实际计算的过程中首先列出样本数据发生概率的表达式（称为似然值），然后分析这个似然值取得最大值时整体模型的概率分布，似然值列出来是个连乘形式，为了方便计算加个log变成连加，求最大值加个-号就是求最小值，最后的形式就是给出的结论（这里具体的数学推导当年没看太懂，学过MLE就很轻松了，就是log likelihood=$P(D|θ)=\sum_{i=1}^n\log p(x_i|\theta)$)；另一种思路是信息论的角度去比较神经网络模型和现实模型的相对熵（KL散度），相对熵就可以表示两个概率模型之间的差距，等于交叉熵减去其中一个模型的熵，由于后者固定，所以交叉熵就可以度量误差，也就是给出的结论形式。  
    后面的梯度下降也是一样了，wj和b更新规则也是和线性回归时梯度下降一样的公式wj = wj - α*(J对wj的偏导)，形式是一样的，上面讨论了区别是f(x)是复合了sigmoid还有误差使用了交叉熵。确认收敛也是一样的方法观察学习曲线比较每次下降的值，什么向量化表示，特征缩放也都是通用的。  
  
#### 多分类问题怎么做  
    答案就是使用Softmax，原理很简单也就是输出层所有节点计算的结果除以所有计算结果的和做一个百分比，更抽象一点说Softmax能够将一组实数值映射成一个离散概率分布。具体使用输入一个实数值，还需要知道同组其他实数值，输出该实数值在这个组中的概率。Softmax也看做是激活函数，但是非常特别的一点就在于它的计算需要把整个一层节点的线性输出都作为输入，因此计算成本会高一些，而且和sigmoid一样映射到0-1比较小容易梯度消失，很少在中间层使用（也可能用，NLP啥的），一般就是输出层多分类场景使用。  
    注意是所有z = wx + b的z，然后都是做$\frac{e^{z_i}}{e^{z_1}+e^{z_2}+e^{z_3}...}$这样，设计成这样的形式（相当于是非线性的归一化而不是用简单的标准归一化都除以sum），我看了下有一些讨论但是都很模糊其实没什么说服力，基本都没说清楚这个问题。自己总结下一方面是可以得到有可解释性的0-1概率形式（不是必要的），z的取值范围是正无穷到负无穷，取指数可以保证值一定是正数，而且保证同一层所有输出的概率合为1，所以加和作为分母，而且做了指数的效果是进一步放大了最大值压制了其他值，可以清楚地根据输出层输出最大的项确认分类，二是在数学上softmax和单分类的sigmoid形式保持了一致，这就只能说是意会了，认为这个形式计算损失函数的时候效果好，而且做反向传播也可以求导。仔细看看softmax的损失函数，会发现当只有1项输出分类时，softmax的形式和sigmoid是一样的  
    损失函数也是使用交叉熵，和二分类是一样的，不过二分类问题应该说是一个特例，结果可能为1/0，比较巧妙一个式子把两种情况考虑了。多分类的话应该是多个输出中有一项为1，单个case的loss=$\sum_{i=1}^{n} y_i\cdot(-\log_{}{p_i})$ ，输出有很多项，这里也就是做了循环累加。  
    那对于softmax函数这样很非线性的怎么求导做反向传播呢？可以先参考下max()函数（取最大值）的求导处理，就是计算时实际取的哪一项输入，就这一项的偏导为1，其他忽略偏导为0，很合理。softmax激活函数也是，由于实际结果为1的项目只有一个，所以实际结果是第i项就用第i项对应节点softmax输出计算loss=-log g(z)，不对为0项预测为1进行惩罚了。  
    另外还有一种是多标签分类问题，就是YOLO做的那种一张图里面判断是否有各种东西识别出来的，多标签multi-label和多类multi-class的问题区分开。可以视为多个二分类问题，理论上也可以视为多个问题用多个神经网络去做，但实际上一个网络就可以高效解决。这里的输出层就不用Softmax了，还是用sigmoid表示多种标签目标存在的概率。  
  
### 过拟合的定义和应对  
    过拟合形象一点很好理解，不过好像还不太好给出严格定义。回归问题的过拟合形象一点理解就是拟合的曲线能够很好的拟合训练集中的点，但是可能曲线为了尽可能贴合训练数据就很诡异，例如多项式次数很高的曲线可以完美经过所有训练集的点，但是这样的拟合曲线歪歪扭扭就没有泛化能力了，任何一个新的点预测结果都可能和实际值有很大差距。分类问题的过拟合形象一点理解就是决策边界为了尽可能区分开训练集中的两类数据，七拐八拐，多项式次数很高可能能完美把所有训练集的点含噪声点全部分好，也是不能泛化了。  
    解决方法：  
* 1.足够多的训练数据，数据不足时还可以做一些数据增强把现有的训练数据数量增加，使用交叉验证应该也是更多利用现有训练数据的方法；  
* 2.减少特征数量，尤其是高阶多项式形式的特征（人工处理不太好办，可以使用自动化的方式舍弃不需要的特征，也是特征工程的一部分）；  
* 3.正则化，不直接移除特征，但是可以把不需要/多项式次数很高的特征的参数设置为接近0，降低权重减少影响，也可以叫做权重衰减weight decay。正则化应该说是机器学习中一个非常重要的部分，为了减少模型的泛化误差基本都会做。机器学习里的正则化regularization和字符串处理中的正则表达式regular expression没有任何关系。除了Andrew课程中提到的这3个，还查到有   
* 4.简化模型，隐含层节点越少越不容易过拟合；  
* 5.早停，梯度下降等训练不一直做到收敛，提前停止不让模型非常拟合训练数据；  
* 6.dropout层随机舍弃数据。具体做法是训练过程，经过BN层时，每一个神经元都只有keep probability的概率保留（或者说1-p的概率丢弃），丢弃的神经元不参与前向、反向传播，相当于改变了网络结构删除一些节点。注意是每次epoch处理一batch数据的时候做一次丢弃，也就是每一轮都随机丢的不一样，训练时是不丢弃的，不能引入随机性导致预测不稳定。那这里训练丢了输出更少，测试不丢输出更大，就引入了问题，解决方法是rescale缩放，原始vanilla版本的dropout的做法是测试的时候所有权重值都乘以保留概率keep probability，但实际现在库里都是inverted版本：训练时为了弥补丢弃神经元对输出的影响，所有权重值都乘以1/keep probability。这样做测试的时候就可以直接忽略dropout层。常见的保留概率是0.5-0.8。  
* 7.BN层。这个很有争议，无法理论上直接证明，但是实际使用中似乎BN层确实起到了某种正则的作用（TODO：有很多种抽象理解的方式可以思考下），效果还可以  
  
广义来说能减少泛化误差的方法都可以叫做正则化，除了针对于w参数的处理，方法5、6、7倒是都能算是正则化。  
    正则化怎么做是一个问题，特征很多但是没法确认哪一个重要哪一个不重要，把哪个调低才对？没法预知，那思路肯定是全部特征的参数都一起处理，但是按经验而言对于绝对值大的参数要调低的更多。有更高的惩罚。数学上的实现也很简单巧妙，直接把损失函数加上一个正则项J = 本来的J + λ/2m * 累加m个(wj^2)，也就是损失函数加上了各个特征参数的大小，优化损失函数的时候就会考虑到特征参数大的要多调小了，这里的λ也是个需要按经验调整的权重值，代表了正则项的惩罚要搞多大，太小了就基本上忽略正则化还是原始方法容易过拟合，太大了就会仅考虑正则化项导致结果是把wj都调为接近0欠拟合。对于b一个常数量做正则化没啥意义不需要。这样wj更新的时候要多减去一个α*λ/m * wj（直接每个wj^2加到损失函数里，求导出来直接就是个一次项，都不存在要链式法则的复杂计算），相当于每次做普通梯度下降之前，wj先乘了个系数收缩。加了这个正则项对于做线性回归和逻辑回归都是一样的操作，没有任何区别。  
    用wj^2做正则项叫做L2范数，如果直接用绝对值之和叫做L1范数正则化，L1范数正则化最后的结果会大部分参数w都为0，是稀疏权重矩阵。这里可以深入学习下。  
  
### 多层神经网络如何做梯度下降？ —— 反向传播  
    有个high level的理解：反向传播传播的是误差，但是不像前向传播那样实际每一层有输出的传播，是“虚拟”的传播，实际没有一个误差值在反过来传播。其实是通过求导的方式分析每一层每一个参数对于最终误差的贡献大小。  
    （从B站西凉阿土伯做一点过度，他整理出了关键点，不过介绍的不够清楚）  
    需要的基础知识，计算图+激活函数+梯度下降+复合函数求导链式法则+张量求导，一共五个基础概念。  
    1.计算图就是个概念而已，从这里引出反向传播的概念。  
        就是数学表达式可以用有向图来表示，节点表示运算，入边表示输入，出边表示输出。从左到右的运算叫做正向传播，这个时候如果从右往左计算从右边输入到左边的反馈，就叫做反向传播。神经网络的数学模型就是计算图的形式，反向传播输入的是输出和实际结果的误差，我理解反向传播回去就可以分析误差的由来，得到每一个点的误差函数，从而可以进行训练调整感知机的w、b参数。所以计算图这里就是理解定义。  
    2.激活函数就是感知机和函数之间的重要区别。  
        对前一段的线性函数输出做一个非线性的变换，这里使得神经网络就可以逼近非线性函数，不然前半部分线性变化无非是矩阵相乘，多少层也不过是线性函数罢了，没有办法去实现复杂问题的分类/回归。（1989年Approximations by superpositions of sigmoidal functions论文证明了即使1个隐含层的sigmoid激活函数神经网络理论上可以拟合任意函数（对比分类问题做万能分类器还是需要2个隐含层，1个隐含层只能做一个convex decision region），1991年Approximation capabilities of multilayer feedforward networks进一步证明了任意激活函数都可以，重要的是多层前馈网络。后续还有很多进一步的数学讨论。简单理解一下，激活函数是一个非线性的函数不是直线，激活函数的自变量wx+b，通过调整输入的参数w b就实现了对于激活函数的x轴平移缩放沿y轴翻转，然后输入下一层神经元时的参数又相当于调整激活函数因变量y轴方向的平移缩放和沿x轴翻转，多个神经元就相当于组合累加不同的激活函数，类似拼图一样将一段一段的激活函数拼接起来就可以组成任意的函数拟合各种曲线，越复杂的曲线需要越多的神经元，实际工程中过多的神经就会导致训练的困难无法找到参数）  
        激活函数的选择很有门道，要根据不同的需求做，先说常用的有等值 sigmoid softmax tanh relu等激活函数，第一点最基本的要求就是需要看输出y的取值范围，例如做0/1二分类肯定是sigmoid，做回归结果有正有负就等值，全是正数而且可能很大就relu。第二点在网络层次深的时候，激活函数导数接近0/大于1在链式法则连乘后，可能导致梯度消失/梯度爆炸，当然梯度爆炸情况比较少，sigmoid导数最大值才0.25还是很容易消失。第三点要考虑计算量，sigmoid就算是还好算的了，当然Relu大于0导数直接为1，小于0导数直接为0更简单（严格而言0点是尖点导数不存在，但实际使用中默认取0/0.5/1就可以了），可以极大降低计算量。目前而言ReLU是用的最多的，sigmoid计算的时候还是麻烦一些而且容易梯度消失用的不多。以sigmoid函数为例，之所以能做激活函数，是因为它具有非常好的数学性质，y=1/(1+e^(-x))，当x=0的时候y=1/2，x正无穷y趋近于1，x负无穷y趋近于0，实现将任意实数映射到0-1的区间。然后sigmoid函数求导之后的形式也不是很复杂，$y'=(1-y)*y$，这个也比较方便计算。另外不同层还可以用不同的激活函数。  
    3.梯度下降法用于寻找调参最优的极值点实现神经网络的最佳拟合。  
        前面Andrew课程部分会分析的比较细。  
    4.复合函数求导链式法则  
        这里对我们学过高数的而言不是啥特别需要提的内容了~当神经网络多层时输出就相当于输入的复合函数，跨层的反向传播要求导，那多层就需要链式法则乘过去。y=g(u),u=f(x),y对于x求导等于g对u求导\*u对x求导。为了推导公式熟练一些，真的要好好练练复合函数求导这里。  
    5.张量求导  
        张量是多维数组的一个统称（TensorFlow的tensor就是张量），0维标量，一维矢量，二维矩阵，三维矩阵数组……都可以称为N维张量。我们平时做的y=f(x)求导，x是一个0层标量，y是一个0维标量。在神经网络中因为节点非常多，要做反向传播的求导一个一个算实在是很难表示。数学上是定义了张量求导的，即因变量y可能是张量，可能是向量或者矩阵有很多，自变量x也可能是张量。常见的还都是标量、矢量、矩阵三种，再高了也不好表示了。张量求导的结果可以表示为雅可比矩阵，以x和y都是矢量为例，y为m维矢量，x为n维矢量，那么雅可比矩阵就是m行\*n列的，第i行j列为yi对xj求偏导的结果。x或者y为矩阵的话那求导结果就是矩阵里面套向量，如果xy都是矩阵，矩阵对矩阵求导那就是矩阵套矩阵，原理都差不多。所以挺好玩的一点是求导是升阶操作，看求导的两个张量本身是几维，结果是维数相加，例如矩阵2求导矩阵2，就是4维。知道了张量求导的形式之后看BP算法数学推导的计算就会比较清楚，神经网络的一层就可以表示为 向量y = 矩阵w * 向量x + 向量b 的形式，计算出dy/dw dy/dx dy/db的张量求导结果就好就计算了。  
    终章---BP算法推导  
        简单一点，用一个3层模型，输入层2个节点，隐含层3个节点，输出层1个节点，把要算的东西都列出来  
            每一层的输出向量 a0 a1 a2，激活函数σ（统一用sigmoid好求导），每一层激活函数前的线性输出向量z1 z2，权重矩阵w1 w2，偏置向量b1 b2  
            先来正向传播，a1 = σ(z1) = σ(w1 * a0 + b)，继续a2 = σ(z2) = σ(w2 * a1 + b2)，可以获得最终的输出  
            有了最终输出a2，和目标y比较可以获得误差，使用MSE最小均方误差为损失函数，L = 累加(y-a2)^2，我们来做反向传播。反向传播的误差值记为ab2 ab1 ab0（aback的意思）  
            反向传播的输入应该不是随便让y和a2做个差，而是损失函数求导的结果，ab2 = L' = y-a2，这里我一开始实在很懵逼，不知道为啥要这么计算ab，后面看到梯度下降更新w、b的时候就恍然大悟。其实理解上的顺序应该是先讲 为了梯度下降更新w、b，要求dL/dw和dL/db，每一层的计算很麻烦，所以为了方便计算，先计算一个反向传播的权值ab，这样形式就会比较干净。课件都是直接先给出ab的计算公式，直接就看不懂为啥要这么做了。  
            接着去按计算图的反向传播公式计算，ab1 = ab2 * (da2/da1)，我一开始不理解da2/da1是什么意义，看到下面就明白了，这就是链式法则求导加上的，放在这里预先算出来，那第1层的w、b就可以直接用ab1计算了比较省事。a2和a1的公式上面介绍了，a2 = σ(z2) = σ(w2 * a1 + b2)，链式法则就是ab1 = ab2 * dσ/dz2 * dz2/da1 = ab2 * dσ/dz2 * w2，同理继续推ab0 = ab1 * (da1/da0) = ab1 * dσ/dz1 * w1  
            算出了正向传播得到误差，反向传播得到各个神经元的误差权值l，l是w和b的函数，就可以用梯度下降更新w、b。w=w + (-1) * (dL/dw) * 学习率，b也是 b = b + (-1) * (dL/db) * 学习率。L前面给出了是L = 累加(y-a2)^2，其中a2就是一个关于w a b的函数，理论上可以对任一一个点的w a b去求导，只是跨层就麻烦一点要链式法则把跨层的导数da2/da1、da1/da0这样的乘上，所以对于每一层的节点而言，dL/dw就可以用已经计算出来的ab来表示，例如第1层就是w = w + (-1) * ab1 * da1/dw * 学习率，ab1 * /da1/dw这个是个核心，展开算有点多，侧面理解下，首先张量点乘（这里都写*号了，应该是点乘）的结果可以认为是阶数相加-2，例如向量1*向量1-2=标量0，向量1*矩阵2-2=向量1。第1层是2个输入，3个节点输出，ab1 * da1/dw = ab1 * da1/dz1 * dz1/dw，其中ab1是1*3向量，da1/dz1是3*3矩阵，dz1/dw是3*2*2的矩阵向量，最后乘出来就是个2*2矩阵，和w1是一致的，也就是代表了w1更新的权值。  
  
### 梯度下降的优化  
#### Batch Size改变：(Full) Batch、mini-batch、SGD  
    首先是降低每轮梯度下降的计算量。全部训练数据都跑一边累加出损失函数然后做梯度下降，称为批量梯度下降BGD；另一个极端就是跑一个训练数据就计算损失函数做梯度下降，称为随机梯度下降SGD；折中的方式就是小批量梯度下降MBGD，跑一小批数据做梯度下降。BGD一定是计算出全局最优，但是样本量大的时候每次计算全部数据才更新一次训练很慢；反之SGD训练非常快，但是不一定是全局最优，而且没法做并行加速。现在实际的应用中基本都是根据情况选择一个批量的值，也就是MBGD，也习惯统一称为SGD了。数学上可以证明对于凸问题而言SGD误差很小，相对提升的训练效率很大。  
    Batch Size也是一个超参数，可能影响最终训练模型的时间和性能。取多大合适没有定值，需要根据数据量和模型大小综合确定。一般而言batch size越小，单epoch训练越快，训练时梯度越不稳定，越容易出现不收敛，因此小批量梯度下降和随机梯度下降的收敛时间都具有不稳定性，在极端情况可能快速收敛也可能长期无法收敛，很难直接判断多大的时候训练耗时最短。  
#### 路径优化：牛顿法、引入冲量Momentum  
    然后是优化梯度下降的路径，更“少走弯路”到达极值点，减少步数（轮次）。可以想到梯度下降求解的梯度仅仅是对于当前点而言，然后沿着这个方向直线行进，那其实就不再是最陡峭的方向了，尤其是SGD走的还不一定是实际梯度方向，所以都有优化的空间。关于学习率的设置，如果一直是固定一个学习率很难手动设置合适，大了不收敛小了训练慢。  
    一个思路是既然按梯度走直线走的远了偏离了最优路径，那升级到二阶，不走直线走曲线看看，会有更接近最优路径的效果。这就是牛顿法，但是多变量二阶导求Hessian矩阵确实计算量大，所以不是很实用，只能说理论上确实更好。  
    另一个思路就还是在走直线一阶的前提下少走弯路。考虑振荡情况，在步长较大的情况下，很可能实际走的路径是在反复振荡前进的。很巧妙的思路就是考虑历史的数据，确认本次w更新差值ΔW(t)i =  ∂J(W(t-1)i)/∂Wi后，实际的更新值V(t) = β*V(t-1) + (1-β)*ΔW(t)，加上这个系数β的方法数学上叫做指数加权移动平均，效果是每次走一步历史数据的影响就*β，实现了越久的数据影响越小，上一步数据影响最大，这样就实现了一定程度上振荡的抵消。这就是动量法Momentum。还有更狠的想法来改进动量法，就是加入预测未来，求偏导的时候输入的点就不是当前点，而是当前点加上历史的冲量，相当于得到惯性往前挪了一点之后的梯度，这样修正出的方向应该就会更接近最优路径，具体计算我看的有点懵，这就叫做Nesterov法加速，也有叫做牛顿动量法的，和二阶导的牛顿法没有关系。  
#### 动态调整学习率  
    再考虑动态调整学习率。理论上越靠近极值点就要越走越慢，所以有固定每次减小学习率的做法，但是这种做法还是不灵活的，其实可以根据一些信息去做自适应的调整。一是Adagrad（adaptive gradient）方法，思路是每个维度上的w的学习率都分开调整，避免在单个方向上一直冲的过快（有点像是归一化的操作），所以某个参数的导数一直比较大的话，就会降低学习率，数学上的实现是学习率每次要除以（历史上所有的ΔW的平方和再开方 + 一个很小的ε避免分母为0），所以某个wi调整的越多学习率就会越小。Adagrad方法对于稀疏特征比较友好（TODO: 这一条很多地方都写了，但是我没有想明白原因，需要研究下）。二是RMSProp方法，这个是Adagrad的进一步改进，因为Adagrad会一直减少学习率，后面就走不动了，如果是有“平台”的情况就会难以完成训练到达机制点。改进也就是历史数据对于学习率的影响不是一直积累，应该近期数据影响大，久远数据影响小，具体的数学实现也是指数加权移动平均，数学式略复杂不详细记述了。  
    动量法是额外加了一个0次项，学习率的调整是对于1次项系数的动态调整，所以这俩方法是可以结合的，RMSProp+Momentum就是目前最常用的Adam方法，虽然具体实现的数学式子已经比较复杂了，但是TensorFlow等库封装的很好直接调用就行。  
  
### 模型评估  
#### 如何评估模型性能？  
##### 训练集：模型性能测试  
    训练数据可以不全都拿来用，分开训练集train set和测试集test set，例如小规模数据集上73开、82开这样划分，大规模数据就看情况找一部分做测试集就行。  
    评估模型性能，回归问题可以比较训练集的误差函数J和测试集的误差函数J（训练的时候应带正则化项，比较的时候不需要）；分类问题当然也可以比较误差函数，但实际上只要跑一遍测试集看分类准确率就很直观。  
##### 验证集：选择确认模型超参数      
    如果还不确定选择什么样的模型比较合适，还需要多个模型都测试看一看效果，那只分为两组是不够的，因为拿测试集来选模型，那本身这个模型就已经是在测试集数据上表现好的了，没有办法客观地评估模型针对新数据的泛化能力。可以622开、811开这样增加一组交叉验证集cross validation set（或者简称验证集validation set/开发集development set）。首先训练集数据训练之后，各个模型在验证集上比较性能，选择性能最好的。所以验证集的作用是确定模型网络结构以及确定模型的超参数（指的是设计模型的参数，例如非线性回归多项式次数，决策树深度等）。然后测试集上再评估模型的泛化能力。  
    深度学习训练的一大难点就是超参数的设置，如果模型不大训练数据量不多，多跑几遍选择最好的超参数当然是理想的，但是现在随着模型规模和训练数据的增加，跑一轮训练都很需要算力，也导致超参数并不好做。  
    但是在训练数据集本身不大的情况下，再分出验证集就很吃数据量了，搞不好就过拟合。所以可以使用k-fold交叉验证的方法。例如做5-fold，就是训练集分为5份，然后依次使用1、2、3、4、5作为验证集跑5次得到平均的结果。这样做的相当于是用更多耗时来交换更多的训练集利用率。  
##### 补充：性能是相对的  
    评估性能好坏的基准不是单纯地看损失函数代表的误差以及绝对的分类准确率，因为对于实际应用而言模型性能好不好是看它有没有用，误差大准确率低不一定不能用。例如一个语音识别系统可能90%的准确率，看起来不高，但是由于训练数据中很多就难以分辨，人类的识别率也就是90%，那这个系统有接近人类的表现，就是很棒的。所以常用的基准包括人类的平均表现水平、已有算法的表现水平、任务需求的表现水平。  
  
#### 模型性能不好怎么办？  
    可能存在的问题有很多，需要多方面的去考虑，然后进行改进。大体是两个方向：数据问题（训练数据少、质量差），模型本身问题（特征选取过多/过少，多项式次数高/低、神经网络节点数多/少）？可以借助一些方法进行诊断，这里介绍bias和variance这两个比较好用的指标。另外介绍随训练数据量增加的学习曲线辅助理解判断数据量不足的情况。  
    1.分析bias和variance判断欠拟合/过拟合  
        bias和variance说起来都是误差，bias译为偏差，代表模型估计值和实际值之间的差距，衡量的是模型准确性，测试、验证集的损失函数应该都可以反映这个，就用J_cv即可表示。variance译为方差，就是统计中代表数据离散程度的方差，衡量的是模型的稳定性，当模型过拟合的时候就会预测结果不稳定应该就会导致方差大（以单特征为例拟合曲线就会歪歪扭扭通过所有训练数据点）。  
            欠拟合 —— bias大，表现是J_cv≈J_train，二者都比较大  
            过拟合 —— variance大，表现是J_train很小，而且J_cv>>J_train  
            很惨的部分过拟合 —— bias和variance都大，连训练数据都搞不好，表现是J_train很大，而且依然J_cv>>J_train  
            模型合适 —— bias和variance都不是很大，J_cv和J_train也都不会很大  
        这里可以回顾下应对过拟合的主流手段——正则化，当正则项系数λ很大的时候，所有的w都会被逐步更新为接近0，导致就剩个偏置b，偏差和方差都大；当正则项系数λ很小为0的时候，相当于没有正则化，会因为多项式特征系数大而过拟合（所以选择正则化系数时，可以设置不同的λ当做不同的模型训练后拿去跑验证集，然后比较找到J最小的）。  
        注意分析bias和variance应该是和基准对比的相对值，如果理论准确率基准就是90%，测试集90%，验证集86%，那此时问题不是欠拟合，而是过拟合。  
    2.理解随着训练数据增加的学习曲线变化，判断是否需要增加训练数据。  
        就是看J_cv和J_train两个曲线，一开始数据量少模型不准确但是很容易强行拟合，所以J_cv很大，J_train很小；随着训练数据越来越多，J_cv逐渐降低，最后趋于平稳也就是模型的性能极限，而J_train反而因为模型拟合能力有限，误差逐步增加，会逐渐升高，最后也趋于平稳。所以J_cv和J_train是在靠近的，结果就是上面第1部分讨论的四种bias&variance情况，最终不一定能基本相等，需要看是否过拟合。  
        欠拟合的情况下，J_cv和J_train最终会基本相等，但大于实际的基准值；过拟合的情况，J_cv平稳后仍远大于J_train，J_train可能小于实际的基准值；部分过拟合的情况，J_cv平稳后的曲线仍远大于J_train，而且J_train本身也很大，远大于基准值。  
        自己总结下应该是考虑了数据量学习曲线问题后，J_cv和J_train还差的比较多，既可能是高variance过拟合也可能是单纯欠拟合情况下数据不足，这里我认为是不好区分的，确认是过拟合可以按之前总结过的方法，继续搞数据/减少特征/增加正则项系数使得J_cv≈J_train；但是数据够了最后的性能仍然和基准有差距欠拟合，应该要优化模型，例如找其他特征/找多项式特征/减少正则项系数（注意没有减少训练数据这种做法，没必要）。  
    看到有论文证明神经网络的本质是多项式回归。但是直接用多项式回归的传统方法做预测可能多项式次数问题不好权衡，还需要验证集去验证分析次数多高才合适。对于深度学习这个问题会简单很多，不需要很小心地去做选择，因为“复杂的神经网络模型在中小型数据集上是低bias的”（TODO：这个我不太理解为什么，大的数据集会有问题么？），按经验来说只要数据量OK，正则化做的好，大的神经网络比小的还好用，不会过拟合。所以拿神经网络就是干，发现J_train不好欠拟合就搞更大的网络，J_cv没J_train好过拟合就搞更多的数据，如此往复可以解决大部分问题。当然很多实际问题也没有这么简单，网络大到一定程度，训练的计算开销就会难以承受，足够多的数据也没有那么好获取，还是需要在现有资源下诊断问题针对性优化。所以再扩展推广一点，整个机器学习开发的通用迭代流程就是：设计架构（模型、数据等）-> 训练模型 -> 诊断问题（分析bias variance error），然后循环往复，诊断出问题再去优化架构，继续训练，继续诊断，直到性能达到需求。  
    再补充一个也是诊断模型性能问题的方法，不算是工具了，就是一个思路，其实很容易想到：  
    3.分析预测错误数据的case（即error analysis）  
        这就是加入人工的判断了，对于预测错误的数据，人工分析总结错误的原因，如果有很多类异常的case，可以整理排序出其中主要的问题优先处理，具体问题具体分析找解决方法。这种思路就比空想一些可能有用的优化更加实际，避免了瞎做一通优化但收效甚微的情况。如果数据量大错误case过多，那就根据人工可以处理的量级随机抽样一部分分析。  
  
#### 训练数据不够怎么办？  
    对比传统的机器学习方法和深度学习，很重要的一个区别就是传统方法更着重于改进模型本身，而深度学习的优化除了模型之外，对训练数据的优化也是很重要的一个部分。要更多训练数据，最直接的当然就是再去人工搜集标记，可以重点搜集error占比大准确率低的类型，但是在确实人力有限，现有训练数据量不足的情况下，还有一些技巧可以有帮助。  
    1.数据增强  
        基于现有训练数据进行修改获得新的训练数据。例如做OCR，对现有的数据做旋转、缩放、对比度调整等，甚至网格化做随机扭曲造成失真；例如做语音识别，对原始的音频加入背景音、一些失真处理；例如做图像分类，可以Mosaic（四张图像拼接一张）、Mixup（随机两个样本按比例混合）、Cutout（随机找区域填0）、CutMix（随机找区域填其他样本块）。这样可以将训练数据翻倍增加，而且这些数据可以很好地提升模型的识别准确率。不过加入纯随机噪声按经验而言没有明显效果。  
    2.数据合成  
        自行合成新的训练数据。例如做OCR，可以基于各种字体+各种变换+各种背景颜色结合在一起随机生成大量训练数据，效率很高，并且和实际去搜集一些文字照片自己标记效果没什么差别。当然需要是能够构造的数据类型，可能还是应用场景有限。  
    3.迁移训练  
        利用其他的充足的训练数据预训练一个模型，然后再修改输出为任务目标，用实际的训练数据进行调优。这么做有效的前提就是认为通过其他训练数据可以获得一些公共的底层特征，所以我理解应该只是部分场景下适用，最典型的就是神经网络做图像分类，用任意的训练数据都可以让前面几层网络提取出一些图像纹理的底层特征（边缘、角点、曲线、基本形状），换了输出层的分类类型也是有意义的。具体到神经网络的处理步骤，以OCR数字图像识别任务为例，可以用任意类别的大量训练数据先训练好一个神经网络，称为监督预训练，然后删除原有的输出层，调整为0-9十个数字的输出层，继续调优有两种方式，一是保持前面的各层参数不变，仅调参输出层，二是以前面各层参数为初始参数，还是整个网络正常训练，按经验而言训练数据较少的时候就前者即可，训练数据多的时候后者的效果会更好。  
        很多监督预训练好的模型可以提供下载，直接用就行，可以极大地降低模型训练所需的数据量，节省时间并且提升性能，开源的良好风气也是深度学习迅速发展的原因之一。注意由于输入层是已经训练好的，输入的数据类型就不能再做更改了，所以以图像识别为例，输入的图像尺寸需要是和预训练模型标准统一的。图像、文本、音频使用同类的训练数据做监督预训练都会有效果，跨类别的数据肯定是没有意义的。  
  
#### 倾斜数据集的误差处理  
    这里是个细节的问题，不知道整理在哪里合适。英文是skew data set，但是没有查到什么相关的讨论，大致记一下。是说分类问题的训练数据，如果各个分类的case不平衡，会引入无法评估误差的问题。举个例子，需要判断是否得病，但是训练数据中99%都没病，1%有病，那模型即使是全部预测没病，对于整体而言也有99%的准确率Accuracy。解决的方法也很简单，误差指标改下分母，以二分类为例，直接分为true positive（预测1正确）false positive（预测1实际0）false negative（预测0世纪1）true negative（预测0正确），定义精确率precision=true positive/true positive+false positve（预测为1的对了多少），定义召回率recall=true positive/true positive+false negative（实际为1的对了多少）。  
    实际应用中，可能无法兼顾precision和positive，需要做权衡。如果是宁缺毋滥的需求，则优先precision；如果是宁可错杀一千不可放过一个，则优先recall。二分类问题可以通过调整输出判断的阈值来实现，例如正常是sigmoid > 0.5是1，那就可以调整这个阈值，阈值高precision高，阈值低recall高。如何权衡可以定义结合precision和recall的指标作为辅助，取算术平均值不太行有极端值是可用的。可以定义F1 score = 1/(0.5*(1/P+1/R)) = 2PR/(P+R)，倒数平均下再倒回来，数学中叫做调和均值harmonic mean，因为调和均值对极小值敏感，所以当precision或recall很小的时候惩罚很高。  
    这样是可以评估模型好坏了，不过我理解模型效果不好的话感觉应该去调整损失函数，让稀少的分类识别错误的惩罚更高，没具体研究了，遇到问题再看吧。  
  
## 机器学习相关补充  
  
### 机器学习应用部署  
    这一部分单独再讨论了。简单而言，通用的做法是封装好机器学习的功能在服务器上运行，对外提供API，输入预测所需数据，输出预测结果，客户端应用请求该API获得结果。涉及到高可用、高并发、成本优化、用户数据存储、监控、模型更新也有一些工程上的技术，可以学习MLOps。  
  
### 机器学习的道德问题  
    机器学习只是从统计数据的角度去分析预测问题，实际应用中可能导致很多影响社会公平、安全的问题，这个需要研究者个人根据道德去判断了，希望是做个好人，不要把机器学习用去伤害人。例如机器学习做招聘、犯罪预测、贷款预测可能导致对性别、肤色、种族、阶级的歧视，强化刻板偏见；机器学习可以伪造图片视频用于诈骗；推荐系统推送煽动性言论；生成虚假的评论信息……  
    学生做学术可能一上来还不用考虑的特别深，但是如果未来有机会做大型应用的话，还是尽量多考虑一下其中可能存在的风险。Andrew提出的一些思路有团队集思广益考虑弱势群体、提前调查特定领域的文献了解相关规则、自己做测试监控可能存在的偏见情况、准备好回滚方案&容灾预案应对突发问题。  
  
## 卷积神经网络  
  
### 引入 —— MLP在处理CV问题时的困难  
    进入计算机视觉相关的部分，输入是图片，首先引入的一个问题就是输入特征过多，例如1000\*1000的RGB图片，输入层1000个节点，那就是30000000三百万个特征输入，输入层的w参数矩阵就是1000\*3000000维的很夸张。特征越多参数越多，训练所需的数据量还有时间就越多，这个是很不现实的。  
    第二个问题是这些输入都是同质的，说不上是直接有意义的特征，可以想到应该是把像素点组合起来才会有意义。  
    第三个问题是需要识别的一些pattern可能在图像的任意位置，即使根据训练数据训练出了可用的网络，实际问题中pattern更改位置就不认识了，需要实现移位不变性shift invariance。  
  
### 卷积 一维-二维-三维  
    卷积是一种特殊的积分变换，有很多种理解的方式。在信号处理与系统中学习线性时不变LTI系统零状态响应的时候引入了零状态响应等于激励信号f(t)和冲击响应h(t)的卷积，忘差不多了应该要回顾下。  
    理解一，数学图形可视化理解一下卷积，输出的函数值实际上是两个函数的对应点“相乘”，不过要把其中一个函数左右翻转过来。  
    理解二，当一个系统输入不稳定但是输出稳定（是个固定的函数）的情况下，用卷积可以计算系统存量。可以按人的进食和消化来理解，已知一个人的随时间变化的进食量函数，以及随时间变化的食物消化剩余量函数，那求解某一时刻人肚子里剩的未消化食物就可以用卷积。（数学公式/f(x)g(t-x)dx，f(x)即进食量，g(x)即消化剩余比例）。  
    理解三，卷积的本质是加权累加，输出并不是仅考虑当前点，而是会考虑周围的点的影响，不同点对输出值的影响大小不同，也就是权重不同。  
    输入为图像时需要二维卷积，形式其实和一维完全不一样，开始我都会怀疑二维卷积为什么叫卷积，可能从理解三的角度更好理解一些。二维卷积计算是首先准备一个卷积核（一个参数固定的矩阵，可以视为一个特定纹理/边缘的过滤器），卷积核“放在”输入图片上对应元素相乘累加，结果填入一个新的矩阵，然后卷积核按设定的步长再Z字形挪动逐步计算，最终获得一个记录卷积输出的新矩阵（可以称为feature map特征图）。这样也可以理解为卷积核确定了周围点对当前点输出的影响权重，相当于一维卷积中的“稳定输出”，这样就符合卷积的概念了。  
    传统的数学、信号处理领域定义的二维卷积应该也和一维要把其中一个函数翻转过来再“相乘”一样，需要对卷积核做一个翻转（和转置不一样，需要先上下翻转再左右翻转），然后再按位点乘，这样就可以实现可交换性（应该是交换律和结合律都行），利于公式的推导变换。但是深度学习中不需要这样严格的做法，反正卷积核是可以学习出来的，目的是提取特征而已，多一步翻转也不必要，严格而言不翻转应该叫做交叉相关cross correlation，不过约定俗称深度学习中就把交叉相关称为卷积。  
    再扩展到三维也是一样的操作，卷积核变成了一个三阶张量。理论上卷积核有三轴的移动会更复杂，但只要卷积核的“高”和输入三阶张量的“高”一致，那仍然是两轴的移动，输出为一个矩阵。实际最常用的场景是彩色图像有RGB三个通道，提升效率一起卷积，卷积核用F\*F\*3的即可。不仅仅是原始图像的多通道，还可以继续扩展，用三维卷积的形式可以同时处理多个特征图，例如第一层用了10个不同的卷积核获得了10个特征图矩阵，那可以将它们叠加成一个“高”为10的三阶张量，一起做卷积。这里的“高”习惯称为深度depth或者通道数number of channels。  
  
### 关于卷积核、步长  
    直观感受卷积核可以看做是一种特定纹理/边缘的检测器，可以想到在遇到和卷积核参数“匹配”的纹理/边缘时会有比较大的输出，就可以反映图像中是否有特定的纹理/边缘，例如常见的垂直/水平边缘检测器，当然也有取平均值平滑图像的平滑卷积核等等。已经有很多已经定义好的实用的卷积核可用，在深度学习中，和MLP中学习w b参数一样，也可以通过反向传播的方法去计算出合适的卷积核。  
    卷积核的尺寸选择，可以认为卷积核越大感受野越大，越可以获得图像的全局特征，但是因为计算量会增加，所以也不能搞得很大。3\*3的就很常见。  
    另外由于直接把卷积核移动范围限制在原始图像内，最后生成的图像的长宽会缩小卷积核边长F-1个像素，而且这样的计算会导致图像边缘像素的“权重”降低，所以通常还会在原始图像周围填充padding一圈(F-1)/2边长的0使得图像不缩小（考虑到要填充的问题一般卷积核不用偶数边长，理论上也可以做不对称的填充不太好）。  
    然后是卷积核每次移动的距离，常用是1，但是可以用其他的，但是要考虑到加上填充能比较好的覆盖原始图像，可以计算下。设图像N\*N，卷积核F\*F，填充边长P，步长S，那么特征图矩阵的边长应该是(N+2P-F)/S + 1，理解起来括号里是可以移动的长度/步长，需要凑成整数，然后初始位置计算一个值再加1。实在是没凑成整数那只能剩一步不走了，不然走出框了没法计算，所以严格来说边长还要做向下取整操作。  
  
### 构建简单的卷积神经网络  
    基于上面的卷积的基础，把输入层用卷积核来构建。同MLP里单个感知机的处理，我们卷积之后获得了一个矩阵，还需要加偏置b（矩阵每个元素都加上就行），以及非线性激活函数（例如Relu，做法也是矩阵每个元素都应用该激活函数）。  
    考虑这里引入的参数，假设3\*3\*3的卷积核一共27个参数，加上1个偏置28个，搞10个卷积核280个，这个参数数量是不大可以训练的，而且可以想到会有不错的泛化能力不容易过拟合，无论原始图片是多大的，不用每个像素对应一个参数。  
     记录一层结构的符号，设层数为l，f^l是卷积核边长，p^l是填充边长，s^l是步长，输入图像大小为$n_H^{l-1} * n_W^{l-1} * n_c^{l-1}$，输出就是l-1都改成l，$n^l = \left \lfloor \frac{n^{l-1}+2p-f}{s} +1 \right \rfloor$。一般卷积核深度会和图像通道数一致，也就是$f^l * f^l * n_c^{l-1}$，这样每个卷积核输出的张量深度等于1，总数就只和卷积核个数有关了，$n_c^l$=卷积核个数。激活函数后l层输出$a^l$就是$n_H^l * n_W^l * n_c^l$的三阶张量。对于每一层结构记录符号没有统一的标准，理论上随便用什么，长宽高什么顺序都是OK的。  
    可以做多个卷积层，一般是逐层缩小输出特征图的长宽尺寸，但是通道数（特征数）可以增加（应该可以理解为逐渐提取出一些高层的特征）。还可以做下面要介绍的池化层。直至输出的特征数量少到方便被处理（例如1000个），就拉成一个一维的向量，和MLP一样全部输入，可以做softmax等去输出预测结果，这就是一般卷积神经网络的最后一层——全连接层（和MLP一样没啥特别的不做额外介绍了）。设计一个卷积神经网络就是要设计这些超参数，包括层数、卷积核大小、填充边长、步长、每一层卷积核数量。  
    再回顾下神经网络中引入卷积是如何解决前面提出的参数过多、输入同质、移位不变三个问题的。可以归纳为参数共享和稀疏连接。想到CV任务中神经网络每一层动辄几千几万的输入和输出，如果使用MLP的话参数多的爆炸，但是引入卷积参数就只是卷积核上一小部分。参数共享的理解就是一个卷积核（特征过滤器）不仅仅对某一部分图像适用，而是对图像中很多区域适用，学习卷积核的参数对很多输入是共享有效的，从而减少了参数冗余，并实现了移位不变。稀疏连接是每一个输出点只和卷积对应的一块原始图像区域连接，原始图像其他点完全不影响，相比全连接减少了非常多的参数。或者换一个思路理解，对于图像局部连接在一起的像素关系紧密，和远方的像素关联少，我们需要“局部感受野”去分辨局部的内容，不需要全局的信息，卷积就实现了这样一个局部感受野。  
  
### 池化  
    光靠卷积还是没法很好解决图像尺寸过大输入过多的问题，直接通过抽样的方法进行压缩。池化pooling这个名字起的我觉得有点迷惑，直接叫抽样sample或者下采样是不是直观一些。做法是把图像划分为多个窗口，例如2\*2 3\*3这样的块，然后每一个块只取一个值（常见的是取最大值max pooling 取平均值mean pooling 也有随机、混合以及很多规则更复杂的方式），这样尺寸直接变为了原来的1/4 1/9，效果非常直接。对比卷积，池化也可以理解为像卷积一样有一个滑动窗口，步长可以小于窗口尺寸，这样有重叠的也是可以的。但是池化操作一般不会做padding没必要，而且一般只缩小图像的高和宽，不会去改变通道数，也就是每个通道上都单独做一样的处理。  
    抽象一点去理解，池化后的特征图和图像压缩一样，会变得模糊，但是在合适的压缩程度之内可以保留重要的图像特征并且大大减少参数量。因为提取的高层特征已经很抽象了，没办法直观说明，只能说实际应用中发现池化效果很好，而且简简单单不会引入待学习的参数。  
    池化后输出的图像尺寸计算和卷积差不多，$n^l = \left \lceil \frac{n^{l-1}+2p-f}{s} +1 \right \rceil$，注意和卷积的区别是需要向上取整，卷积可以加填充，边缘的走不到不好计算卷积就算了，但是池化不做填充，走过了元素不足依然可以取池化。  
    池化层存在的问题是池化做了下采样多输入对单输出了，乍一看直接求导似乎不好做，就做不了反向传播。但这个问题也很好解决，常见的最大值池化就额外记录下最大值位置，反过去最大值一对一就可以求导了；平均值池化就是多个输入平均了一下是可以求导的。  
  
### 关于CNN训练  
    方法和MPL一样，随机初始化卷积核参数，然后梯度下降去调参直至收敛。但是卷积核这里的计算还是要复杂一些，暂不深入分析。  
  
### CNN的发展与经典网络结构  
    多了解一些实用的网络结构及其发展变化对于自己设计神经网络很有帮助。  
#### leNet 1998奠基网络  
    Yann LeCun教授1980年代就已经开始研究了，开始是小的卷积网络做一些简单任务，他非常相信这个东西随着未来算力发展是有用的。后面他加入贝尔实验室，有了更好的设备和大型数据集，开始着手设计更加复杂的网络，设计出的leNet-1网络在手写文字识别任务中取得了非常好的效果，迅速得到商业应用（银行支票识别等），1998年发表论文介绍了leNet-5，基本形成了后面CNN网络的雏形，成为了奠基论文。（卷积+池化）\*n + 全连接的基础结构仍然是主流，但其中仍有一些过时内容，包括使用的一些激活函数已经被淘汰，池化后加激活函数的方法也不用了，还有一些复杂的提升计算效率的方法也不需要了。  
    输入32\*32的图像。共6层，  
    第一层5\*5的6个卷积核输出28\*28\*6；然后第二层2\*2的池化输出14\*14\*6（原始的应该用的平均池化）；  
    第三层卷积规则有点复杂，用了16个卷积核，但不是16个5\*5\*6的卷积核，而是分了4组，第一组是5\*5\*3的卷积核，从6个特征图中卷积连续3个的组合（如012 123，共6个），第二组是5\*5\*4的卷积核，从6个特征图中卷积连续4个的组合（如0123 1234，共6个），第三组也是5\*5\*4的卷积核，从6个特征图中卷积两两隔开的组合（0134 1245 2350，共3个），第四组就是5\*5\*6的卷积核1个，最后输出10\*10\*16；第四层还是2\*2池化，输出5\*5\*16；  
    第五层5\*5\*16的120个卷积核，直接卷出1\*1\*120个结果；第六层就是全连接层，120个输出组成一个向量输入，84个神经元，激活函数使用tanh；最后第七层是输出层，10个神经元，对应0-9手写数字识别结果，不过当时没有softmax，用的是RBF径向欧氏距离函数来判断最后的结果，输出越接近0说明距离越近就是结果，现在淘汰了。  
#### AlexNet 2012重新带火深度学习  
    leNet之后深度学习在一些复杂任务重也有小规模的一些研究，但是都没有很好的效果，又进入低谷。2009年李飞飞教授发布了大型图像数据库ImageNet，并开启了ILSVRC竞赛比拼图像识别的任务。2012年ALexNet以16.4%的错误率取得冠军，并且领先第二名10%，从此以后深度学习又进入了一个火热的阶段。  
    Alex Krizhevsky等设计。感觉确实复杂了许多，主要的升级有：Relu激活函数；dropout层应对过拟合；使用步长小于池化核的重叠池化，全部最大池化；LRN层进一步强化大的神经元响应，提升泛化能力（但是后续研究发现意义不大）；使用CUDA GPU加速，显存有限使用2块GPU平分存储（硬件首先但是做的比较麻烦不是通用的做法）；做了数据增强减少过拟合；小批量梯度下降+冲量+权重衰减。  
    输入227\*227\*3的图像，接着5层（卷积+池化）层，2层全连接层，1层输出层。  
    第一层11\*11\*3的96个卷积核，步长为4比较大，激活函数Relu，输出55\*55\*96；接着做了一个局部响应归一化LRN，具体公式不计了，不过我不知道为什么强化响应大的神经元输出可以提升模型泛化能力，另外这里还有把55\*55\*96分为2个55\*5\*48从而放到2块GPU处理的操作；第一层3\*3的池化，步长为2，输出两个27\*27\*48；  
    第二层卷积5\*5\*48的256（一组128）个卷积核加填充2步长1，保持边长不变，输出27\*27\*256；继续第二层LRN；第二层也是3\*3的池化，步长为2，输出13\*13\*256；  
    第三层光卷积不池化了，3\*3\*128的卷积核384（一组192）个，填充1步长1保持边长不变，输出13\*13\*384；  
    第四层也是光卷积不池化，3\*3\*192的384（一组192）个卷积核，填充1步长1保持边长不变，那输出还是13\*13\*384没变；  
    第五层卷积3\*3\*192的256（一组128）个卷积核，填充1步长1保持边长不变，输出13\*13\*256；第五层又开始做池化了，3\*3池化步长2，输出6\*6\*256；  
    第六层全连接层，不过和leNet的第五层一样是同体积的卷积，直接用6\*6\*256的4096个卷积核获得1\*1\*4096个输出；并且后面做了dropout，随机断开4096个神经元中某些的连接，可以防止过拟合；  
    第七层全连接层，输入4096的一个向量，输出还是4096，也是做了dropout，神奇，做了两层难道会更强一些？  
    第八层输出层，1000个输出代表1000个分类列表，softmax。  
    一起有大约65w神经元，6千万个参数。  
#### VGG 2014第二名  
    虽然没有比过第一名，但是这个网络性能也是不错的，堆到了十多层（经典的是VGG16和VGG19），并且结构是比较标准简洁的，做的很“规整”，基本都是几层3\*3的卷积再来一层2\*2池化，卷积核数量也是一直翻倍增加到512，没有搞各式各样的卷积核和池化。作者提出了一些有益的经验，包括LRN层作用不大；提升网络深度性能提升；3\*3的卷积层多来几层就相当于扩大了感受野，例如3\*3两层就是5\*5，三层就是7\*7，并且参数量更少。  
    由于此时还没有引入BN层，所以16/19层的训练其实就已经很困难很需要技巧了，作者使用的方法是先训练11层的，这个还是能收敛。然后在其中再增加层级。  
#### GoogLeNet 2014第一名  
    谷歌做的名字致敬了LeNet，而且引入inception结构的名字即盗梦空间电影的名字，台词有we need to go deeper也表明了设计的目标有点意思。  
    层数直接拉到了22层，非常创造性的一点是引入了并行连接，不只是串行的做卷积了，设计了一个inception的结构，输出结果是四路卷积合起来的结果，可以做1\*1 3\*3 5\*5，甚至还可以做池化（需要步长为1并加填充不减少尺寸），这样就灵活地解决了卷积核尺寸选择的问题。控制填充使得输出长宽不变都是一样的，那就把通道数叠加在一起就行。为了减少参数数量，每一路卷积最后用的是1*1的卷积核实现降维，还可以在卷积前做1*1卷积降维（可以叫做瓶颈层）。  
    1\*1的卷积核听起来很奇怪，对于通道数为1的情况当然没有用，但是卷积神经网络中一般通道数都很多，1\*1的卷积核其实输入的是一个向量，多放几个1\*1卷积核就和MLP的一层一样了是全连接的，可以从“网中网”的角度来理解，输入不同通道的同一点作为特征，所以是有用的，按经验而言只要通道数控制合理，就能省计算并且对网络性能影响小。控制1\*1卷积核的数量就可以调整通道数，即使1*1卷积层输入输出的通道数一样也是有意义的，毕竟增加了激活函数增加了非线性拟合能力更强。  
    inception v1版本还有5\*5的卷积核，计算成本实在是太高了，v2版本就移除改成2层3\*3了，并且提出了Batch Normalization（BN）方法，通过归一化解决梯度消失/爆炸的问题，简单来说的思路就是每一层都根据一部分训练数据归一化输入值（经典的就是归一化为均值为0，方差为1的标准正态分布），使其分布在能让激活函数输出在比较合适的区间，缓解梯度消失/爆炸，从而使得增加层数更容易收敛。TODO: BN层的具体实现研究下。2017年还更新了一版Xception。  
#### Resnet 2015第一名  
    Resnet创造性地引入了残差块的结构，对于网络深度增加后的退化问题有非常好的效果，极大地增加了神经网络可行的深度。神经网络层次深了首先的问题就是梯度消失无法训练，这个做BN层可以解决。但是实际测试中发现即使做深了性能依然没有优于浅层的网络，作者分析是深层网络难以训练。  
残差块的原理比较好理解，就是既然层数多了之后之前层的影响减少了，那就增加一条路径让当前层可以跳过相邻层直接影响后面的层，这个结构可以称为shortcut或者skip connection。  
    不同层之间可能维度不同，前面层的输出没法直接输入后面层，维度匹配在是引入的一个新问题。有两种方式，一是乘以一个矩阵升维降维就行（张量点乘阶数-2，所以三阶张量乘个矩阵还是三阶），说实验发现通过一些复杂的中间处理学习得到一个特别的矩阵或者干脆固定一个矩阵不足的部分填0效果差不多，就不用做的很复杂，直接用zero padding矩阵；二就是用1\*1的卷积核调整个数升维降维，效果好一点，但是引入额外的参数要训练。TODO: 残差网络维度匹配具体怎么做还是要再研究下。  
    一般而言每一层就做卷积就行了，但是层数很多的情况下计算量有点大，所以还有一个先降维再升维的优化方法，是每一层先做1\*1卷积降维，然后3\*3卷积，再1\*1卷积升维回去，这样卷积的时候计算量就会比较少。  
    数学理论上理解为啥残差网络有用，需要理解恒等映射（就是f(x)=x一个不改变自变量的函数），我没太理解……感觉大概的意思是理论上层数越多，神经网络拟合的函数就越多，但是需要保证每一层能拟合的函数都是包含上一层能拟合的函数的并集才可以（称为嵌套网络），不然本来第10层能拟合的，加到50层反而跑偏了输出不了第10层的结果。残差网络直接把前面层的输出叠加到后面层，就保证了后面层的拟合范围更大更厉害，例如当l-2层的残差块叠加l层时，如果l层本身的w参数为0的时候就保持了和l-2层一样，从而实现恒等映射。  
  
#### DenseNet 2017 CVPR Best Paper  
Densely Connected Convolutional Networks  
其实算是极端化的ResNet，发现Skip Connections能带来更好的效果后，DenseNet是直接把所有层都两两连接起来，设有L层，则从第一层入口到第L层出口共有L+1个连接点，每个连接点和另外L个相连，总共连接数就是$L(L+1) / 2$。每一层的输入都是自身输入加上前面所有层的输入。  
优势是彻底不担心梯度消失了，每一层的特征都能有效使用，理论上可以达到最佳的准确率。但直观感觉这样会引入大量网络参数，而实际DenseNet的设计中大大减少了通道数（卷积核数），和下面MobileNet一样使用深度可分离卷积减少参数来那个，最终参数量是小于ResNet的。  
  
#### MobileNet 2017为移动设备运算降低计算量  
为了能够在计算性能不足的移动设备上跑神经网络模型而设计，引入了depthwise separable convolution深度可分离卷积，可以减少计算量。  
简单计算一些，一个卷积层需要计算的乘法次数等于卷积核的尺寸（长宽高$F \times F \times N_{cin}$）\*卷积核计算次数（即输出特征图矩阵的尺寸$N_{out} \times N_{out}$，可以算出来边长$N_{out} = \frac{(N+2P-F)} {S} + 1$）\*卷积核个数$N_{cout}$。为了减少乘法计算的次数，深度可分离卷积的做法是分两步，先做分层做深度卷积depthwise convolution，即所有通道不一起卷积了，例如原始图像3个颜色通道，分别用3个矩阵卷积核，输出矩阵就是3个通道。然后做逐点卷积pointwise convolution，1\*1\*3的卷积核任意调整维数。逐点卷积的乘法次数计算方法一样，也是卷积核尺寸（固定$1 \times 1 \times N_{cout}$）\*输出特征图矩阵的尺寸  
严格计算下，深度可分离卷积的乘法次数除以原始卷积乘法次数为$\frac{深度卷积乘法次数 \times 逐点卷积乘法次数}{原始卷积乘法次数} = \frac{F \times F \times N_{cin} \times N_{out} \times N_{out} + N_{cin} \times N_{cout} \times N_{out} \times N_{out}}{F \times F \times N_{cin} \times N_{out} \times N_{out} \times N_{cout}}$  
化简下即乘法计算量是原始卷积方式的$\frac {1}{N_{cout}}+\frac{1}{F^2}$，当用了很多卷积核输出通道数很多的情况下可以大大减少乘法运算数量，可以想到本来每个卷积核都要做\*3通道的乘法，现在公共用一样的深度卷积结果，虽然要增加多一步逐点卷积，但是1\*1\*3的卷积运算量级还好）  
为了更方便的给模型“瘦身”进一步降低计算量，引入两个超参数Width multiplier和Resolution multiplier。做法很简单就是等比例地降低通道数和图像尺寸，定义Width multiplier为α用于降低通道数，$N_{cin}$和$N_{cout}$降低为$\alpha N_{cin}$和$\alpha N_{cout}$。定义Resolution multiplier为ρ，特征图尺寸N降低为$\rho N$，常见的就是224、192、160、128这样的取值。  
MobileNet v1的结构就是前面先做了13层的深度可分离卷积，后面还是正常的池化全连接。实验证明效果不错，在性能达到之前各类卷积网络的情况下，运算量和参数量大大降低。  
2019年又更新MobileNet v2版本，主要有2点更新。一是讨论了relu激活函数存在的问题，涉及流形学习等概念不深入分析了，大致的结论是低维度的情况下relu会丢信息，所以要先升维再卷积。具体实现是每一层的深度可分离卷积之前增加了expansion扩展操作，一次扩展+深度可分离卷积合在一起称为一个瓶颈层bottleneck。扩展操作也就是用$1 \times 1 \times N_{cin}$的卷积来进行升维，例如通道数\*6，本来输入3个通道，搞18个1\*1\*3的卷积核就是18通道了。然后到了逐点卷积的时候又会去降低通道数，例如18个通道再变回3个。注意1\*1的卷积就不加relu激活函数了，而且为了避免计算复杂，用的relu6（即x超过6之后限制f(x)最大值为6）。二是引入了残差网络的残差连接结构，这个正是在resnet还很火热的时期很好理解，解决梯度消失/爆炸问题，特别的一点是传统残差网络连接的每一层block是为了减少计算量先降维再升维的，这里为了解决relu损失信息问题又深度可分离卷积减少计算量是先升维再降维的，作者起名叫inverted residual block。MobileNet v2一共用了17层的瓶颈层，瓶颈层这样的设计实验证明是有效的。  
  
#### EfficientNet 2019可灵活缩放大小的网络  
因为实际运行神经网络的设备计算能力可能是千差万别的，理论上大而深的网络付出更多计算开销能够有更高的准确率，反之牺牲一些准确率降低计算开销，如果能够灵活调整网络的大小，就可以适用于更多设备。  
调整神经网络大小可以考虑对r-图像分辨率，d-神经网络深度，w-神经网络宽度的调整，但是如何配合调整这3者并取得较好的性能是个问题。EfficientNet主要解决了这一问题。  
这个应该说是“最小最好”的经典CNN模型，如果要搭一些简单CV应用的话很推荐基于它做。应该是经典的YOLOv5版本就是基于EfficientNet的。  
  
## 支持向量机SVM  
做分类任务的传统方法，主要是做二分类任务，当然也是可以做多分类的，会复杂一些。  
SVM核心的思路是间隔最大化。这个间隔不严谨地说就是搞两个板子分开两类数据，在不分类错的情况下让两个板子距离越大越好，但是严谨地数学语言定义这个间隔还是挺麻烦的……TODO: 后面再补充定义吧。假设确实两类数据是线性不可分的没有办法分隔开，那就是做软间隔，允许少量误分类样本的情况下找最大的间隔。软间隔最大化的数学定义也是有点麻烦。  
接着不严谨的说法，两块板子应该是正好贴到两类点的，这两块板子正中间搞块板子就定义为决策超平面，两边的板子叫做上下超平面。  
如果点分不开，很可能是需要升维之后在更高维度的空间中找决策超平面。但是升维的计算是比较复杂的，可以使用kernel trick的方法，在计算量一定的情况下也可以获得高维度下的差异度，从而找到决策超平面。  
  
## 序列模型Sequence Model  
从定义来说只要输入/输出至少一方是序列数据（文本、音频、视频等）就是序列模型，例如音频-文字的语音识别、类型编号-音频的音乐生成、文本-整数的文本评价、DNA序列-位置信息的基因标定、文本-文本的机器翻译、视频-类别的行为识别……  
用一个简单的例子来标注下介绍序列模型时的一些符号。名字实体识别的任务，找出句子中的人名，有多组标签数据。输入数据是一个英文句子，第i组含有$T_x^{(i)}$个词，第t个词用$x^{(i)<t>}$表示；对应的第i组输出是一个$T_y^{(i)}$长度的0/1序列（这个任务中$T_x=T_y$），代表第t位的词是不是人名，用$y^{<t>}$表示。这样x和y都是时间序列了。  
关于词的表示，一般不会一个字母一个字母的ASCII码输入，而是转换为一个向量。词向量化有很多种方式，例如最简单的按字典顺序one-hot编码，或者用一些复杂的预训练好的映射方式。  
### 递归神经网络RNN  
提出的很早，1982年就有雏形了，1990年正式提出但是没解决梯度消失问题应用很有限。  
考虑直接用普通的MLP来做序列问题。例如上面的从句子中识别人名的例子，当然是可以做输入层$T_x \times$词向量维数 个节点直接把整个句子输入进来，输出层$T_y$个节点代表每个词是否是人名。但这只能说对于这个特殊的任务的一个特殊的输入可以做。一方面是任务类型有很多，每一个输入数据和输出数据的长度不一定是固定的，提前固定了输入层和输出层节点数是不灵活的；另一方面是神经网络学习的结果是位置相关的了，一个人名的词汇出现在句子的任何地方都是人名，这里希望有类似卷积那样位置无关的效果。  
另一种方式，我们输入层只一次输入一个词，然后输出层一个节点给出这个词是否是人名的结果，这样当然简单了，但是又完全没有考虑词在句子中的位置和上下文，成为了单词分类器了，这也不厉害。  
#### RNN网络结构  
仍然考虑输入层一次只输入一个词的神经网络，想象一下把这个神经网络复制无数个（一个网络是一个2D的图了，现在想像成一层一层叠着的3D图），第一个网络处理$x^{<1>}$输出$y^{<1>}$，然后让第一个网络的隐含层的输出$a^{<1>}$，同时传入第二个网络的隐含层作为一个额外的输入，通过这样一个很简单的连接，就使得处理第一个词的记忆可以传入第二个词的处理中。  
直观感觉下这样不是太复杂了吗？有多长的序列就要搞多少个网络，这怎么可能？实际操作中我们认为每一个网络的参数都是一样的，即是一个有两路输入（$x^{<t>}$和上一层的记忆$a^{<t-1>}$）和两路输出（$y^{<t>}$和往下一层传的记忆$a^{<t>}$）的网络，只是要一次又一次迭代去计算出从第1个到第$T_x$个输入输出了。由于有了两路输入和两路输出，那现在网络的参数w就有了3组（$w_{aa},w_{ax},w_{ya}$，注意a是两路输入决定的，但y就直接根据隐含层输出a决定），参数b就有了两组（$b_a,b_y$）  
可以发现在这样的计算过程中，前面输入的词会对后面输入的词有影响，但是后面输入的词无法影响到前面的词。（后面也有双向递归的复杂RNN）  
RNN中输出a的激活函数一般使用tanh（也可能使用ReLu），输出y的激活函数还是看任务的需求，分类问题还是用sigmoid、softmax。  
另外隐含层的输出其实就是很多论文里会提到的hidden state隐藏状态，抽象一点理解每一步的hidden state就是包含了之前所有时间的信息，也是RNN“记忆”的来源。Andrew老师用的是符号a表示，很多地方用的是h表示。  
#### RNN正向传播  
两路输出两个式子，为了形式好看一点记录简化后的形式  
$a^{<t>}=g(w_a[a^{<t-1>},x^{<t>}]+b_a)$  
其中$w_a$是$w_{aa}$和$w_{ax}$向量拼在一起  
$\hat{y}^{<t>}=g(w_ya^{<t>}+b_y)$  
其中$w_y$就是$w_{ya}$的简写了  
#### RNN反向传播  
损失函数L等于每一步输出$y^{<t>}$的损失总和  
反向传播也是递归回去计算，和正向传播的递归计算反向，即从最后的$T_y$步一直倒回去传播到第1步。（TODO：没有细算，有空推算下流程理解下）  
可以理解为“基于时间的反向传播”（Andrew老师很喜欢这个名字，我觉得好像也平平无奇）  
#### 支持不同输入输出的RNN  
前面的例子是输入输出序列同长度的最简单情况，但是复杂的输入输出不等长或者有一侧不是序列的情况，都可以通过修改基本的RNN架构来实现。  
一对一的RNN其实就是个MLP不需要讨论。  
例如文本评价任务，输入文本序列，输出单个整数分值，多对一，可以前面的递归中仅计算隐含层输出a但是不输出y，仅仅在最后一次$T_x$的迭代输出一个y就可以了。  
例如风格音乐生成任务，输入一个风格编号，输出音频序列，一对多。那也是仅第一次迭代输入一个x，后面递归就相当于只有一路来自上一次迭代隐含层的输出$a^{<t-1>}$。  
例如机器翻译任务，输入输出是不等长的文本序列，多对多。可以完全把RNN做成两个部分，前一部分迭代仅按$T_x$长度输入x，后一部分仅按$T_y$长度输出y，就实现了x和y的长度可以无关联。  
#### RNN做语言模型  
TODO  
语言模型的核心是预测特定词汇的出现概率。每次选择下一个最大概率出现的词汇就可以生成文本序列了。  
根据语料库训练。  
训练好的RNN语言模型就确定了特定词汇的出现概率，具体概率可以通过采样的方式估计。  
#### RNN的问题——梯度消失  
RNN有“记忆”，但是并不擅长于长期记忆，几次迭代之后前面的信息对于当前隐含层的输出影响就很小了。其中的原因就和单个神经网络做深了梯度消失很难反向传播一样，RNN迭代也拉长了反向传播的路径。  
梯度爆炸很好处理，只要clipping一下限制最大值（或者缩放一下）就可以解决。但是梯度消失不行。按经验来看，RNN的递归次数也和CNN的层数一样，需要看数据集、激活函数、网络结构、是否引入预训练等信息，大部分在10次以内，多的也就50次以内。  
  
### 长短期记忆网络LSTM  
1997年提出的。  
可以理解为在RNN的基础上增加了另外一条长期记忆记录链路，抽象一点可以理解为一个日记本记录着长期的记忆，而原本的RNN视为短期记忆链。我们将这条链路的输出记为$c^{<t>}$。每次递归，隐含层$a^{<t-1>}$输出后，会根据$a^{<t-1>}$去删除一些$c^{<t-1>}$中的旧记忆，再往里添加一些新记忆，得到$c^{<t>}$，然后反过来$a^{<t>}$在增加一路输入$c^{<t-1>}$  
换一种“门控”的角度来理解，引入了长期记忆链路，可以理解为增加了三个门：遗忘门（根据 当前时刻输入+上一时刻隐藏状态 判断长期记忆遗忘哪些）、输入门（控制 当前时刻输入+上一时刻隐藏状态 进入长期记忆）、输出门（控制 长期记忆 输出到当前时刻隐藏状态）。即通过门控机制来控制需要额外输入、遗忘、输出什么。  
某种意义上可以认为LSTM的长期记忆线是和ResNet的Skip Connections有相似之处，都是拉通了更加直接的传播路径，将浅层网络（早期网络）的信息传播到深层网络（后期网络）。  
图示反正看起来是挺复杂的，不过手算一下递归计算公式感觉还可以：  
$\begin{pmatrix} i \\ f \\ o \\ g \end{pmatrix} = \begin{pmatrix} \sigma \\ \sigma \\ \sigma \\ \tanh \end{pmatrix}W\begin{pmatrix} h_{t-1} \\ x_t \end{pmatrix}$  
$c_t=f\odot c_{t-1}+i\odot g$  
$h_t=o\odot \tanh(c_t)$  
比较迷惑的是说好是三个门input-i forget-f output-o，还多了个g，这是gate gate，其实是和i一起组成input门，因为i是sigmoid激活函数控制比例，g是tanh激活函数可以控制input写多少到长期记忆。i f o g计算都差不多只是分了4组参数，还比较好记。然后$\odot$这个是对应元素相乘，因为是做每个元素变一下比例，不然向量相乘含义就不对了。  
  
### 门控循环单元GRU  
2014年提出的，Gated Recurrent Unit。  
GRU和LSTM一样都是基于RNN变形的，懂了LSTM的结构之后看GRU的结构会更简单。  
LSTM效果很强，但是由于引入了三个门，参数量增加了很多，训练和推理的难度和开销都打了很多。  
GRU某种意义上可以看做是LSTM的简化，名字就是门控，实际只引入了两个门，更新门和重置门。其中更新门的作用可以看做LSTM中输入门和遗忘门的合并，实际做法很简单就是在上一时刻隐藏状态和当前时刻输入混合的时候调整权重（记旧的多一点还是记新的多一点，获得一个简化版本的长期记忆）。而重置门和输出门的的做法也不太一样，非常简单就是控制着一个权重向量，在“简化版本的长期记忆”与“短期记忆”上一时刻隐藏状态混合的时候控制权重。  
这里又是Andrew老师课程用的符号不太一样自己选的……我用通用表示了，记录$h_t$是隐含层输出的隐藏状态，$r_t$是经过更新门后的输出，$z_t$是经过重置门后的输出，$x_t$是当前时刻输入。然后因为要通过重置门多算一个中间状态（“简化版本的长期记忆”），我们记为候选隐藏状态$\hat h_t$。  
一共四个公式，首先经过更新门和重置门的输出形式是一样的，不过接下来作用不一样  
$z_t=\sigma(W_z \cdot [h_{t-1},x_t])$  
$r_t=\sigma(W_r \cdot [h_{t-1},x_t])$  
$\hat h_t=tanh(W \cdot [r_t* h_{t-1},x_t])$  
$h_t=(1-z_t)* h_{t-1}+z_t* \hat h_t$  
我理解GRU没有显式地引入另外一路长期记忆链路，往下递归传递的还是仅有一个隐藏变量，但是把隐藏变量分了两路做做处理实现了类似长期记忆+短期记忆的效果（一个拆成两个用），能在RNN的计算效率与LSTM优秀的长期记忆之间取了个平衡，还是挺抽象的。  

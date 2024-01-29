* Q-Bench: A Benchmark for General-Purpose Foundation Models on Low-level Vision  
Weisi Lin老师团队，做的用GPT-4V多模态大预言模型MLLM去试试低级视觉感知、低级视觉描述、整体视觉质量评估，还做了个挺大的数据集，实现了量化对比人类和GPT-4V视觉感知能力，广受好评  
* ImagenHub: Standardizing the evaluation of conditional image generation models  
滑铁卢大学Tiger AI实验室，做文本和图像生成的  
做了多个不同图像生成任务的评估数据集来评估图像生成质量，评价不错，可以了解下看能不能用  
* Quality Diversity through Human Feedback  
人类评价的质量多样性指标，也是类似图像质量一样比较抽象难以量化的指标，可以指导强化学习等场景，不要做唯一标准有利于生成模型。  
图像生成重视内容真实性、多样性的指标，可以了解下，可能和图像质量评估有相通可借鉴的地方  
* Perceptual Context and Sensitivity in Image Quality Assessment: A Human-Centric Approach  
这个似乎也是视觉敏感性相关，但是创新性评价不高。提到的对比学习、random masking、hard negative mining难例挖掘？、模型蒸馏的方法都是组合的比较成熟的方法。我还不太了解都需要学习下。  
* Disentangling the Link Between Image Statistics and Human Perception  
非常视觉理论的文章，我没有看的太懂。在讨论图像统计（具体说是图像概率）和感知灵敏度之间的关系。  
没太看懂图像概率是啥东西，是信息论里的那个，但是具体的计算方法应该是不容易的，文章中使用了PixelCNN++来估计。感知灵敏度我也没看懂是什么……直观感觉和图像质量评估是有点相关的  
评价也不高，评委们指出了没有研究人类感知情况而是使用了感知距离指标的问题，导致实际意义不强，而且实验也做的不深  
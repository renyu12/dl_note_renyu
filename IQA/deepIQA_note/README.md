S. Bosse, D. Maniry, K. -R. Müller, T. Wiegand and W. Samek, "Deep Neural Networks for No-Reference and Full-Reference Image Quality Assessment," in IEEE Transactions on Image Processing, vol. 27, no. 1, pp. 206-219, Jan. 2018, doi: 10.1109/TIP.2017.2760518.
2018年TIP经典的用深度学习做NR&FR IQA的论文。
提出了新的做NR&FR IQA的网络结构，同一个网络加上参考输入就可以做FR，去掉参考输入就做NR。由于数据量不够，输入不是整个图像，而是每个图像平均采样32个32\*32的补丁汇集起来增加了数据量。另外还考虑到不同补丁的重要性不同，另外引入了一个网络判断不同补丁对质量影响的权重值加入预测。

# deepIQA

This is the reference implementation of [Deep Neural Networks for No-Reference and Full-Reference Image Quality Assessment][arxiv].
The pretrained models contained in the models directory were trained for both NR and FR IQA and for both model variants described in the paper.
They were trained on the full LIVE or TID2013 database respectively, as used in the cross-dataset evaluations. This evaluation script uses non-overlapping 32x32 patches to produce deterministic scores, whereas the evaluation in the paper uses randomly sampled overlapping patches. 

> usage: evaluate.py [-h] [--model MODEL] [--top {patchwise,weighted}]
>                   [--gpu GPU]
>                   INPUT [REF]

## Dependencies
* [chainer](http://chainer.org/)
* ~~scikit-learn~~
* ~~opencv~~

## TODO 
* add training code
* add cpu support (minor change)
* ~~remove opencv and scikit-learn dependencies for loading data (minor changes)~~
* ~~fix non-deterministic behaviour~~

[arxiv]: http://arxiv.org/abs/1612.01697

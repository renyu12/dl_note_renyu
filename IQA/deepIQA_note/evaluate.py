#!/usr/bin/python2
import numpy as np
from numpy.lib.stride_tricks import as_strided    # renyu: 比较难理解的一个自定义步幅实现滑动窗口采样的方法，可以学习下，平均采局部补丁使用

import chainer    # renyu: 用的是比较早期的一个深度学习框架chainer，看起来也还不错
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers

import argparse
import six
import imageio
import numbers
     
# renyu: 读取另外两个文件里定义的FR和NR网络模型
from nr_model import Model
from fr_model import FRModel

# renyu: 提取局部补丁的操作，默认应该是32*32彩色补丁，平均一张图提取32个
def extract_patches(arr, patch_shape=(32,32,3), extraction_step=32):
    arr_ndim = arr.ndim   # renyu: 这个维数应该是输入图像的张数

    if isinstance(patch_shape, numbers.Number):
        patch_shape = tuple([patch_shape] * arr_ndim)
    if isinstance(extraction_step, numbers.Number):
        extraction_step = tuple([extraction_step] * arr_ndim)

    patch_strides = arr.strides

    slices = tuple(slice(None, None, st) for st in extraction_step)
    indexing_strides = arr[slices].strides

    patch_indices_shape = ((np.array(arr.shape) - np.array(patch_shape)) //
                           np.array(extraction_step)) + 1

    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))

    patches = as_strided(arr, shape=shape, strides=strides)    # renyu: 用的as_strided方法采样，具体的原理和参数设置要学习下TODO
    return patches

# renyu: 调用时可以命令行设置下待评估图片路径，参考图片路径（为空就是NR），已训练好模型的路径，是否使用补丁加权，GPU ID
parser = argparse.ArgumentParser(description='evaluate.py')
parser.add_argument('INPUT', help='path to input image')
parser.add_argument('REF', default="", nargs="?", help='path to reference image, if omitted NR IQA is assumed')
parser.add_argument('--model', '-m', default='',
                    help='path to the trained model')
parser.add_argument('--top', choices=('patchwise', 'weighted'),
                    default='weighted', help='top layer and loss definition')
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID')
args = parser.parse_args()


chainer.global_config.train = False
chainer.global_config.cudnn_deterministic = True


FR = True
if args.REF == "":    # renyu: 是否FR根据参考图片路径是否为空判断
     FR = False

if FR:
     model = FRModel(top=args.top)
else:
     model = Model(top=args.top)


cuda.cudnn_enabled = True
cuda.check_cuda_available()
xp = cuda.cupy
serializers.load_hdf5(args.model, model)
model.to_gpu()

# renyu: 提取参考图补丁，做转置。这里调整了图片补丁格式（行，列，颜色）为（颜色，行，列）应该是chainer框架的输入格式，pytorch也是一样的
if FR:
     ref_img = imageio.imread(args.REF)    # imageio读进来的图片数据格式需要了解下TODO
     patches = extract_patches(ref_img)
     X_ref = np.transpose(patches.reshape((-1, 32, 32, 3)), (0, 3, 1, 2))

# renyu: 提取待评估图补丁，做转置
img = imageio.imread(args.INPUT)
patches = extract_patches(img)
X = np.transpose(patches.reshape((-1, 32, 32, 3)), (0, 3, 1, 2))


y = []
weights = []
batchsize = min(2000, X.shape[0])    # renyu: 限制一轮最多跑2000张图片
t = xp.zeros((1, 1), np.float32)    # renyu: TODO: 这里传了个0是什么操作？
for i in six.moves.range(0, X.shape[0], batchsize):
     X_batch = X[i:i + batchsize]
     X_batch = xp.array(X_batch.astype(np.float32))

     # renyu: model.forward就是定义好的运行模型预测的方法
     if FR:
          X_ref_batch = X_ref[i:i + batchsize]
          X_ref_batch = xp.array(X_ref_batch.astype(np.float32))
          model.forward(X_batch, X_ref_batch, t, False, n_patches_per_image=X_batch.shape[0])
     else:
          model.forward(X_batch, t, False, X_batch.shape[0])

     y.append(xp.asnumpy(model.y[0].data).reshape((-1,)))
     weights.append(xp.asnumpy(model.a[0].data).reshape((-1,)))

# renyu: TODO: 这里结果的形式没看明白，好像损失函数就是取得MSE，所以就都加起来除以补丁个数？
y = np.concatenate(y)
weights = np.concatenate(weights)

print("%f" %  (np.sum(y*weights)/np.sum(weights)))

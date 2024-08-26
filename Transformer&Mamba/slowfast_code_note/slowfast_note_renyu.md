# 安装  
好久没配环境一开始基本的Pytorch环境配置就踩坑了，AutoDL11.8的CUDA 需要安装对应版本的Pytorch，还是按官网的一些组合去装吧  
```  
# 基本环境  
conda create --name slowfast1 python=3.10  
conda init bash && source /root/.bashrc  
conda activate slowfast1  
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda==11.8 -c pytorch -c nvidia  
  
# facebook搞的一个AI库，有一些常用的功能，包括参数&FLOPs统计  
git clone https://github.com/facebookresearch/fvcore  
pip install -e fvcore  
  
# 好像是处理COCO数据集的库？感觉用不到吧  
git clone https://github.com/cocodataset/cocoapi  
pip install -e cocoapi/PythonAPI  
  
# Facebook的CV库，集成了一些目标检测、语义分割模型啥的  
# 可能有一些依赖项问题，可以先pip装后面的一些依赖包  
git clone https://github.com/facebookresearch/detectron2 detectron2_repo  
pip install -e detectron2_repo  
  
# Facebook SlowFast库主代码  
# 后面发现这里直接装有无数问题……这个库比较早20年就建了，后面还一直小加点东西，也没有发过release版本……直接搞的最新版代码是跑不起来的很坑……建议git reset --hard 旧一点的版本  
git clone https://github.com/facebookresearch/slowfast  
cd slowfast  
python setup.py build develop  
  
# 24.8.13新版本改了run_net.py文件的库路径，不兼容旧版了……测试时用的24.8.14的新版本，好多问题，本来想checkout单个文件，发现越调越多，应该要整体回退下比较好……  
#git checkout adc55ca38a62a8bf3899a2d7c2dfbc9f9eee1ade slowfast/utils/ava_eval_helper.py  
  
  
# 回退22.8.4试试  
git reset --hard 99a655bd533d7fddd7f79509e3dfaae811767b5c  
git checkout adc55ca38a62a8bf3899a2d7c2dfbc9f9eee1ade tools/run_net.py  
  
# pytorchvideo直接pip装少文件……这个也得源码安装  
git clone https://github.com/facebookresearch/pytorchvideo.git  
pip install -e pytorchvideo  
  
# 安装文档很旧，列了一些包，没照着那个装，自己一个一个报错试出来的，大概是下面一些  
pip install opencv-python av simplejson pandas pillow psutil scipy scikit-learn  
```  
# 测试启动  
不出所料是跑不起来的，坑也不少  
## 配置问题（显示GPU错误）  
```  
# 这个是文档里提供的测试启动命令，测试一直报显卡问题，检查GPU没有问题，后面发现是后面的命令行参数根本没有被解析，只读了config文件的配置（默认8卡），我的解决方法是直接都写到yaml配置文件里，不要弄命令行参数  
python tools/run_net.py --cfg configs/Kinetics/C2D_8x8_R50.yaml NUM_GPUS 1 TRAIN.BATCH_SIZE 8 SOLVER.BASE_LR 0.0125 DATA.PATH_TO_DATA_DIR /auto-fs/data/renyu_kinectics-2  
# 几个配置都写到yaml里，然后启动命令改为  
python tools/run_net.py --cfg configs/Kinetics/C2D_8x8_R50_test.yaml  
```  
## PyTorch2兼容问题（显示torch._six错误）  
```  
#不兼容新版Pytorch问题，from torch._six import int_classes as _int_classes报错ModuleNotFoundError: No module named 'torch._six'。不需要去导入了，改成  
int_classes = int可以解决  
# 涉及两个文件build/lib/slowfast/datasets/multigrid_helper.py和slowfast/datasets/multigrid_helper.py，本来是检查1.8版本之后用int，但是2.0版本没兼容，改一下就好  
elif TORCH_MAJOR >= 2:  
    _int_classes = int  
```  
  
  
```  
#跑起来有个只读numpy数组不能直接转tensor的报错信息比较烦，加个.copy()拷贝吧……  
vim slowfast/datasets/decoder.py  
video_tensor = torch.from_numpy(np.frombuffer(video_handle, dtype=np.uint8).copy())  
```  
  
# 改VQA回归任务  
## label从int改float  
```  
# 代码部分复用Kinetics数据集处理，但label要从int改为float  
  
loss部分slowfast/models/losses.py要有mse "mse": nn.MSELoss,  
# 配置项中设置  
MODEL.LOSS_FUNC: mse  
```  
但是这里还是有问题没有跑通，似乎是反向传播时dtype为double（float64），但是网络需要float32，这里没能定位问题，有待进一步调试看是哪里发生的不一致  
## 加载预训练模型  
```  
# 可以在github库里下载到一些模型的预训练模型，然后用对应的配置文件，里面会有加载预训练模型的配置项，包括  
TRAIN:  
  ENABLE: False  
    
# 但是SlowFast没有提供Finetune的代码可能还得自己搞下，加载预训练参数然后从头训练按说也可以  
TRAIN.CHECKPOINT_FILE_PATH: /auto-fs/data/videomamba_pretrain_models/I3D_8x8_R50.pkl 用预训练模型微调的配置  
```  
  
## 单卡GPU调整  
```  
NUM_GPUS: 1  
BATCH_SIZE: 64 看显存去适当降低吧  
```  
## 其他配置  
```  
BASE_LR: 0.0125 小一点  
DATA.PATH_TO_DATA_DIR: renyu_kinetics-2  
```
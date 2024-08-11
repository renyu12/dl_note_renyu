import argparse
import os

import numpy as np

import torch
import torch.nn
from torchvision import transforms

from model import UGC_BVQA_model

from utils import performance_fit

from data_loader import VideoDataset_images_with_motion_features



def main(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # renyu: 读取预训练模型，注意这里的预训练模型是Spatial Feature提取的resnet50加上最后的两层MLP回归模型，对应论文框图的下边和右边部分
    #        可以看UGC_BVQA_model.py代码，不是原始的resnet50
    if config.model_name == 'UGC_BVQA_model':
        print('The current model is ' + config.model_name)
        model = UGC_BVQA_model.resnet50(pretrained=False)


    model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
    model = model.to(device)

    # load the trained model
    print('loading the trained model')
    model.load_state_dict(torch.load(config.trained_model))


    # renyu: 直接支持的测试集包括LSVQ测试集、KoNViD和Youtube UGC，预先准备好提取的帧和SlowFast网络输出特征就可以
    if config.database == 'LSVQ_test':
        datainfo_test = 'data/LSVQ_whole_test.csv'
        videos_dir = os.path.join(config.data_path, 'LSVQ_image')
        feature_dir = os.path.join(config.data_path, 'LSVQ_SlowFast_feature/')
    elif config.database == 'KoNViD-1k':
        datainfo_test = 'data/KoNViD-1k_data.mat'
        videos_dir = os.path.join(config.data_path, 'konvid1k_image')
        feature_dir = os.path.join(config.data_path, 'konvid1k_SlowFast_feature/')
    elif config.database == 'youtube_ugc':
        datainfo_test = 'data/youtube_ugc_data.mat'
        videos_dir = os.path.join(config.data_path, 'youtube_ugc/youtube_ugc_image')
        feature_dir = os.path.join(config.data_path, 'outube_ugc/youtube_ugc_SlowFast_feature/')

    # renyu: 这里是2D Frame要做的变换处理，也不算是很高清，resize到520*520再中心crop到448*448
    transformations_test = transforms.Compose([transforms.Resize(520),transforms.CenterCrop(448),\
        transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
  
    testset = VideoDataset_images_with_motion_features(videos_dir, feature_dir, datainfo_test, \
        transformations_test, config.database, 448, config.feature_type)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
        shuffle=False, num_workers=config.num_workers)

    with torch.no_grad():
        model.eval()
        label = np.zeros([len(testset)])
        y_output = np.zeros([len(testset)])
        videos_name = []
        # renyu: DataLoader返回的第一项是8帧Frames，第二项是从本地文件读取的预处理好的motion特征numpy数组，正好对应model的输入
        for i, (video, feature_3D, mos, video_name) in enumerate(test_loader):
            print(video_name[0])
            videos_name.append(video_name)
            video = video.to(device)
            feature_3D = feature_3D.to(device)
            label[i] = mos.item()
            outputs = model(video, feature_3D)

            y_output[i] = outputs.item()
        
        val_PLCC, val_SRCC, val_KRCC, val_RMSE = performance_fit(label, y_output)
        
        print('The result on the databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(\
            val_SRCC, val_KRCC, val_PLCC, val_RMSE))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--database', type=str)
    parser.add_argument('--train_database', type=str)
    parser.add_argument('--model_name', type=str)

    parser.add_argument('--num_workers', type=int, default=6)

    # misc
    parser.add_argument('--trained_model', type=str, default='ckpts')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--feature_type', type=str)
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--gpu_ids', type=list, default=None)

    
    config = parser.parse_args()

    main(config)




import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
from tqdm import tqdm
from collections import OrderedDict
from model.utils import DataLoader
from model.final_future_prediction_shanghai import *
from utils import *
import glob
import argparse

def MNADEval(model=None):
    parser = argparse.ArgumentParser(description="DMAD")
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
    parser.add_argument('--h', type=int, default=256, help='height of input images')
    parser.add_argument('--w', type=int, default=256, help='width of input images')
    parser.add_argument('--c', type=int, default=1, help='channel of input images')
    parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
    parser.add_argument('--dim', type=int, default=512, help='channel dimension of the memory items')
    parser.add_argument('--msize', type=int, default=1000, help='number of the memory items')
    parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
    parser.add_argument('--dataset_type', type=str, default='shanghai', help='type of dataset: ped2, avenue, shanghai')
    parser.add_argument('--dataset_path', type=str, default='your default dataset path', help='directory of data')
    parser.add_argument('--model_dir', type=str,  default='./modelzoo/shanghai.pth', help='directory of model')

    args = parser.parse_args()

    torch.manual_seed(2020)
    os.environ["CUDA_VISIBLE_DEVICES"]= "0"
    torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance
    test_folder = args.dataset_path+args.dataset_type+"/testing/frames"

    # Loading dataset
    test_dataset = DataLoader(test_folder, transforms.Compose([
                transforms.ToTensor(),
                ]), resize_height=args.h, resize_width=args.w, time_step=4, c=args.c)

    test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size,
                                shuffle=False, num_workers=args.num_workers_test, drop_last=False)

    # Loading the trained model
    if model is None: # if not training, we give a exist model and params path
        model = convAE(args.c, 5, args.msize, args.dim)
        try:
            model.load_state_dict(torch.load(args.model_dir).state_dict(),strict=False)
        except:
            model.load_state_dict(torch.load(args.model_dir),strict=False)
        model.load_bkg(get_bkg(args.w))
        model.cuda()
    else:
        model = convAE(args.c, 5, args.msize, args.dim)
        model_dir = './exp/shanghai/log/temp.pth'
        try:
            model.load_state_dict(torch.load(model_dir).state_dict())
        except:
            model.load_state_dict(torch.load(model_dir))
        model.load_bkg(get_bkg(args.w))
        model.cuda()

    labels_list = np.load('./data/frame_labels_'+args.dataset_type+'.npy')

    videos = OrderedDict()
    videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
    for video in videos_list:
        video_name = video.split('/')[-1]
        videos[video_name] = {}
        videos[video_name]['path'] = video
        videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
        videos[video_name]['frame'].sort()
        videos[video_name]['length'] = len(videos[video_name]['frame'])

    label_length = 0
    list1 = {}
    list2 = {}
    list3 = {}

    print('Evaluation of', args.dataset_type)

    # Setting for video anomaly detection
    for video in sorted(videos_list):
        video_name = video.split('/')[-1]
        label_length += videos[video_name]['length']
        list1[video_name] = []
        list2[video_name] = []
        list3[video_name] = []
    
    label_length = 0
    video_num = 0
    label_length += videos[videos_list[video_num].split('/')[-1]]['length']

    model.eval()
    with torch.no_grad():
        for k,(imgs, view_idx) in enumerate(tqdm(test_batch)):

            if k == label_length-4*(video_num+1):
                video_num += 1
                label_length += videos[videos_list[video_num].split('/')[-1]]['length']
            
            vidx = Variable((view_idx[:, 0,0,0]+330).long()).cuda()
            imgs = Variable(imgs).cuda()
            outputs = model.forward(imgs[:, 0:12],vidx, True)
            err_map = model.loss_function(imgs[:, -3:], vidx, *outputs, True)
            latest_losses = model.latest_losses()

            list1[videos_list[video_num].split('/')[-1]].append(float(latest_losses['err1']))
            list2[videos_list[video_num].split('/')[-1]].append(float(latest_losses['err2_']))

            ##############norm with cv2.findContours################
            mask = outputs[-1].squeeze(0).squeeze(0).cpu().numpy()
            mask = np.where(mask > 0.7, 1, 0).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max3 = 0
            for k in range(len(contours)):
                if cv2.contourArea(contours[k]) > 13:
                    mask_ = cv2.drawContours(np.zeros_like(mask), contours, k, 255, -1).astype(np.float32)
                    area = mask_.sum()
                    err = (mask_ * err_map.squeeze(0).cpu().numpy()).sum() / area
                    if err > max3:
                        max3 = err
            list3[videos_list[video_num].split('/')[-1]].append(max3)

    # Measuring the abnormality score and the AUC
    anomaly_list1 = []
    anomaly_list2 = []
    anomaly_list3 = []

    for video in sorted(videos_list):
        video_name = video.split('/')[-1]
        anomaly_list1 += conf_avg(np.array(anomaly_score_list_inv(list1[video_name])), n_conf="robust+")
        anomaly_list2 += conf_avg(np.array(anomaly_score_list_inv(list2[video_name])), n_conf="robust+")
        anomaly_list3 += conf_avg(np.array(anomaly_score_list_inv(list3[video_name])), n_conf="robust+")

    np.savez("./exp/shanghai/log/res_list.npz", anomaly_list1, anomaly_list2, anomaly_list3, 1 - labels_list)

    anomaly_list1 = conf_avg(np.array(anomaly_list1), 55, "average")
    anomaly_list2 = conf_avg(np.array(anomaly_list2), 55, "average")
    anomaly_list3 = conf_avg(np.array(anomaly_list3), 55, "average")
    
    hyp_alpha = [0.2, 0.4, 0.6]

    comb = np.array(anomaly_list1) * hyp_alpha[0] + np.array(anomaly_list2) * hyp_alpha[1] + np.array(anomaly_list3) * hyp_alpha[2]
    accuracy = filter(np.array(conf_avg(comb, 55)), 1 - labels_list, test_folder)

    print('The result of ', args.dataset_type)
    print('AUC: ', accuracy * 100, '%; (alpha = ', hyp_alpha, ')')

    return accuracy * 100


if __name__=='__main__':
    MNADEval()

#     data = np.load("./exp.bak/res_list_shanghai.npz")

#     anomaly_list1 = conf_avg(np.array(data['arr_0']), 55, "average")
#     anomaly_list2 = conf_avg(np.array(data['arr_2']), 55, "average")
#     anomaly_list3 = conf_avg(np.array(data['arr_3']), 55, "average")

#     hyp_alpha = [0.2, 0.6, 0.4]
    
#     comb = np.array(anomaly_list1) * hyp_alpha[0] + np.array(anomaly_list2) * hyp_alpha[1] + np.array(anomaly_list3) * hyp_alpha[2]

#     accuracy = filter(np.array(conf_avg(comb, 55)), data['arr_5'][0], "..\\..\\shanghai\\testing\\frames")
#     print('AUC: ', accuracy * 100, '%; (alpha = ', hyp_alpha, ')')

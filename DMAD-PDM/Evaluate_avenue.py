import numpy as np
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
from collections import OrderedDict
from model.utils import DataLoader
from model.final_future_prediction_avenue import *
from utils import *
import glob

import argparse

def MNADEval(model=None):
    parser = argparse.ArgumentParser(description="DMAD")
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
    parser.add_argument('--h', type=int, default=256, help='height of input images')
    parser.add_argument('--w', type=int, default=256, help='width of input images')
    parser.add_argument('--c', type=int, default=1, help='channel of input images')
    parser.add_argument('--dim', type=int, default=512, help='channel dimension of the memory items')
    parser.add_argument('--msize', type=int, default=420, help='number of the memory items')
    parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
    parser.add_argument('--dataset_type', type=str, default='avenue', help='type of dataset: ped2, avenue, shanghai')
    parser.add_argument('--dataset_path', type=str, default='your default dataset path', help='directory of data')
    parser.add_argument('--model_dir', type=str,  default='./modelzoo/avenue_.pth', help='directory of model')
    args = parser.parse_args()

    torch.manual_seed(2020)
    os.environ["CUDA_VISIBLE_DEVICES"]= '0'
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
        ifTraining = False
        model = convAE(args.c, 5, args.msize, args.dim)
        # use for torch < 1.6.0 zip files
        try:
            model.load_state_dict(torch.load(args.model_dir).state_dict(),strict=False)
        except:
            model.load_state_dict(torch.load(args.model_dir),strict=False)
        model.cuda()
    else:
        ifTraining = True

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
    grad_list_x, grad_list_y = [], []
    with torch.no_grad():
        for k,(imgs, _) in enumerate(test_batch):

            if k == label_length-4*(video_num+1):
                video_num += 1
                label_length += videos[videos_list[video_num].split('/')[-1]]['length']

            imgs = Variable(imgs).cuda()

            outputs = model.forward(imgs[:, :12],True)
            model.loss_function(imgs[:, -3:], *outputs, True)
            latest_losses = model.latest_losses()

            list1[videos_list[video_num].split('/')[-1]].append(float(latest_losses['err1']))
            list2[videos_list[video_num].split('/')[-1]].append(float(latest_losses['mse2']))
            list3[videos_list[video_num].split('/')[-1]].append(float(latest_losses['grad']))

            dif = ((imgs[:, -3:]-outputs[0]).abs().squeeze(0).cpu().numpy().transpose((1,2,0))*127.5).astype(np.uint8)
            cv2.imwrite("./exp/"+str(k)+'_0.jpg', dif)
            grad_list_x.append(outputs[6][:,:,:,1].cpu().numpy().astype(np.float16)) # 8bit is ok

    if not ifTraining:
        exp_offset = np.concatenate(grad_list_x, axis=0)
        exp_offset = exp_offset / np.abs(exp_offset).max()
        np.save("./exp/offset8.npy", (exp_offset * 127).astype(np.int8))
        print("please used post-precessing to remove static novel instance and evaluate the final auc")
        return

    # Measuring the abnormality score and the AUC
    anomaly_list1 = []
    anomaly_list2 = []
    anomaly_list3 = []

    for video in sorted(videos_list):
        video_name = video.split('/')[-1]
        anomaly_list1 += anomaly_score_list_inv(list1[video_name])
        anomaly_list2 += anomaly_score_list_inv(list2[video_name])
        anomaly_list3 += anomaly_score_list_inv(list3[video_name])

    anomaly_list1 = np.asarray(anomaly_list1)
    anomaly_list2 = np.asarray(anomaly_list2)
    anomaly_list3 = np.asarray(anomaly_list3)

    hyp_alpha = [0.2, 0.4, 0.6]

    comb = np.array(anomaly_list1) * hyp_alpha[0] + np.array(anomaly_list2) * hyp_alpha[1] + np.array(anomaly_list3) * hyp_alpha[2]
    accuracy = roc_auc_score(y_true=1 - labels_list, y_score=comb)

    print('The result of ', args.dataset_type)
    print('AUC: ', accuracy*100, '%; (alpha = ', hyp_alpha, ')')

    return accuracy*100

if __name__=='__main__':
    MNADEval()



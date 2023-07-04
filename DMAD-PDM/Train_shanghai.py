import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
import copy
from collections import OrderedDict
from model.utils import DataLoader
from model.final_future_prediction_shanghai import *
from utils import *
import Evaluate_shanghai as Evaluate
import argparse

def MNADTrain():
    parser = argparse.ArgumentParser(description="DMAD")
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs for training')
    parser.add_argument('--h', type=int, default=256, help='height of input images')
    parser.add_argument('--w', type=int, default=256, help='width of input images')
    parser.add_argument('--c', type=int, default=1, help='channel of input images')
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate') # why not 3e-4 XD
    parser.add_argument('--dim', type=int, default=512, help='channel dimension of the memory items')
    parser.add_argument('--msize', type=int, default=1000, help='number of the memory items')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
    parser.add_argument('--pin_memory', default=True, help='pinned memory for faster training, use more cpu')
    parser.add_argument('--dataset_type', type=str, default='shanghai', help='type of dataset: ped2, avenue, shanghai')
    parser.add_argument('--dataset_path', type=str, default='H:\\info_tech\\AI\\DataSet\\anomaly_detection\\', help='directory of data')
    parser.add_argument('--exp_dir', type=str, default='log', help='directory of log')
    parser.add_argument('--log_type', type=str, default='realtime', help='type of log: txt, realtime')
    args = parser.parse_args()

    np.random.seed(2021)
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

    train_folder = args.dataset_path+args.dataset_type+"/training/frames"
    test_folder = args.dataset_path+args.dataset_type+"/testing/frames"
    
    if "bkg" not in os.listdir('./'):
        os.mkdir("./bkg")
        gen_bkg(train_folder, "./bkg/")
        print("background images has been generated, please re-run the Train.py")
        return

    if "bkg_" not in os.listdir('./'):
        os.mkdir("./bkg_")
        gen_bkg(test_folder, "./bkg_/")
        print("background images has been generated, please re-run the Train.py")
        return

    bkg = get_bkg(args.w)
    print("loading a ",bkg.shape[0],"-views background template")

    # Loading dataset
    train_dataset = DataLoader(train_folder, transforms.Compose([
                 transforms.ToTensor(),
                 ]), resize_height=args.h, resize_width=args.w, time_step=4, c=args.c)
    train_batch = data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, drop_last=True, pin_memory=args.pin_memory)

    # Model setting
    model = convAE(args.c, 5, args.msize, args.dim, bkg=bkg)
    optimizer = torch.optim.AdamW([{'params': model.encoder.parameters()},
        {'params': model.decoder.parameters()},
        {'params': model.offset_net.parameters()},
        {'params': model.vq_layer.parameters()},
        {'params': model.bkg, "lr": 10*args.lr},]
        , lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    model.cuda()

    # Report the training process
    log_dir = os.path.join('./exp', args.dataset_type, args.exp_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    orig_stdout = sys.stdout
    f = open(os.path.join(log_dir, 'log.txt'),'w')
    if args.log_type == 'txt':
        sys.stdout = f

    # Training
    early_stop = {'idx' : 0,
                  'best_eval_auc' : 0}
    # record = [0]
    log_interval = 100
    loss_dict = model.latest_losses()
    losses = {k + '_train': 0 for k, v in loss_dict.items()}
    epoch_losses = {k + '_train': 0 for k, v in loss_dict.items()}

    for epoch in range(args.epochs):
        model.train()
        for j,(imgs, view_idx) in enumerate(train_batch):
            imgs = Variable(imgs).cuda()
            vidx = Variable((view_idx[:, 0,0,0]).long()).cuda()
            outputs = model.forward(imgs[:,0:12], vidx)

            optimizer.zero_grad()
            loss = model.loss_function(imgs[:,-3:],vidx,  *outputs)
            loss.backward()
            optimizer.step()
            ########################################
            latest_losses = model.latest_losses()
            for key in latest_losses:
                losses[key + '_train'] += float(latest_losses[key])
                epoch_losses[key + '_train'] += float(latest_losses[key])

            if j % log_interval == 0:
                for key in latest_losses:
                    losses[key + '_train'] /= log_interval
                loss_string = ' '.join(['{}: {:.6f}'.format(k, v) for k, v in losses.items()])
                print('Train Epoch: {epoch} [{batch:5d}/{total_batch} ({percent:2d}%)]  {loss}'
                             .format(epoch=epoch, batch=j * len(imgs),
                                     total_batch=len(train_batch) * len(imgs),
                                     percent=int(100. * j / len(train_batch)),
                                     loss=loss_string))
                print("best: ", early_stop['best_eval_auc'])
                for key in latest_losses:
                    losses[key + '_train'] = 0

        scheduler.step()

        print('----------------------------------------')
        print('Epoch:', epoch+1)
        print('----------------------------------------')
        torch.save(model, os.path.join(log_dir, 'temp.pth'))
        score = Evaluate.MNADEval(True)

        if epoch%10 == 0:
            if score > early_stop['best_eval_auc']:
                print('With {} epochs, best score is: {}'.format(epoch+1, early_stop['best_eval_auc']))
                early_stop['best_eval_auc'] = score
                torch.save(model, os.path.join(log_dir, 'model.pth'))
                break
        print('With {} epochs, auc score is: {}, best score is: {}'.format(epoch+1, score,early_stop['best_eval_auc']))

    print('Training is finished')
    sys.stdout = orig_stdout
    f.close()


if __name__=='__main__':
    MNADTrain()

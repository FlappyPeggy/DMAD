import torch
from dataset import get_data_transforms
import numpy as np
import random
import os
from resnet import wide_resnet50_2
from de_resnet import de_wide_resnet50_2
from dataset import MVTecDataset
from test import evaluation
from torch.nn import functional as F

ifgeom = ['screw', 'carpet', 'metal_nut'] # include geometrical changes AND discrimination of feature similarity is weak

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def loss_fucntion(a, b):
    #mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        #print(a[item].shape)
        #print(b[item].shape)
        #loss += 0.1*mse_loss(a[item], b[item])
        loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),
                                      b[item].view(b[item].shape[0],-1)))
    return loss

def loss_concat(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a[0].shape[-1]
    for item in range(len(a)):
        #loss += mse_loss(a[item], b[item])
        a_map.append(F.interpolate(a[item], size=size, mode='bilinear', align_corners=True))
        b_map.append(F.interpolate(b[item], size=size, mode='bilinear', align_corners=True))
    a_map = torch.cat(a_map,1)
    b_map = torch.cat(b_map,1)
    loss += torch.mean(1-cos_loss(a_map,b_map))
    return loss

def eval(_class_, rec, root, ckpt_path, ifgeom):
    print(_class_)
    image_size = 256
    mode = "sp"
    path = 'wres50_' + _class_ + ('_I.pth' if mode=="sp" else '_P.pth')
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vq = mode == "sp"

    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    test_path = root + _class_
    
    ckp_path = ckpt_path + path
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, bn, offset = wide_resnet50_2(pretrained=True, vq=vq)
    encoder = encoder.to(device)
    bn = bn.to(device)
    offset = offset.to(device)
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    encoder.eval()
    
    print("evaluating")
    ckp = torch.load(ckp_path)
    decoder.load_state_dict(ckp['decoder'], strict=False)
    bn.load_state_dict(ckp['bn'], strict=False)
    offset.load_state_dict(ckp['offset'], strict=False)
    auroc_px, auroc_sp = evaluation(offset, encoder, bn, decoder, test_dataloader, device, _class_, mode, ifgeom)

    return auroc_sp if mode=="sp" else auroc_px


if __name__ == '__main__':
    root_path = "your dataset root path"
    ckpt_path = "your ckpt path"
    item_list = ['capsule', 'cable','screw','pill','carpet', 'bottle', 'hazelnut','leather', 'grid','transistor', 'metal_nut', 'toothbrush', 'zipper', 'tile', 'wood']
    rec = []
    for i in item_list:
        rec.append(eval(i, rec, root_path, ckpt_path, ifgeom=i in ifgeom))
    print(rec)

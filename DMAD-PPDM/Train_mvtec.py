import torch
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
import cv2
from resnet import wide_resnet50_2
from de_resnet import de_wide_resnet50_2
from dataset import MVTecDataset
from test import evaluation
from torch.nn import functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.backends.cudnn.benchmark = True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

ifgeom = ['screw', 'carpet', 'metal_nut']

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

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

def train(_class_, root='./mvtec/', ckpt_path='./ckpt/', ifgeom=None):
    print(_class_)
    epochs = 400
    learning_rate = 0.005
    batch_size = 8
    image_size = 256
    mode = "sp"
    gamma = 1
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vq = mode == "sp"
    print(device)

    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    train_path = root + _class_ + '/train'
    test_path = root + _class_
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    ckp_path = ckpt_path + 'wres50_' + _class_ + ('_I.pth' if mode=="sp" else '_P.pth')
    train_data = ImageFolder(root=train_path, transform=data_transform)
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, bn, offset = wide_resnet50_2(pretrained=True, vq=vq, gamma=gamma)
    encoder = encoder.to(device)
    bn = bn.to(device)
    offset = offset.to(device)
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    encoder.eval()

    optimizer = torch.optim.AdamW(list(offset.parameters())+list(decoder.parameters())+list(bn.parameters()), lr=learning_rate, betas=(0.5,0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        offset.train()
        bn.train()
        decoder.train()
        loss_rec = {"main":[0],
                    "offset":[0],
                    "vq":[0]}
        for k, (img, label) in enumerate(train_dataloader):
            img = img.to(device)
            _, img_, offset_loss = offset(img)
            # check img
            # if k%20 == 0:
            #     cv2.imwrite("./exp/"+str(k)+".jpg", cv2.cvtColor(((img_[0].detach().cpu().numpy()*np.array([[[0.229]], [[0.224]], [[0.225]]])+np.array([[[0.485]], [[0.456]], [[0.406]]]))*255).astype(np.uint8).transpose((1,2,0)), cv2.COLOR_RGB2BGR))
            inputs = encoder(img_)
            vq, vq_loss = bn(inputs)
            outputs = decoder(vq)
            main_loss = loss_fucntion(inputs, outputs)
            loss = main_loss + offset_loss + vq_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_rec["main"].append(main_loss.item())
            loss_rec["offset"].append(offset_loss.item())
            try:
                loss_rec["vq"].append(vq_loss.item())
            except:
                loss_rec["vq"].append(0)
        print('epoch [{}/{}], main_loss:{:.4f}, offset_loss:{:.4f}, vq_loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_rec["main"]), np.mean(loss_rec["offset"]), np.mean(loss_rec["vq"])))
        if (epoch + 1) % 10 == 0:
            auroc = evaluation(offset, encoder, bn, decoder, test_dataloader, device, _class_, mode, ifgeom)
            torch.save({
                'offset': offset.state_dict(),
                'bn': bn.state_dict(),
                'decoder': decoder.state_dict()}, ckp_path)
            print('Auroc:{:.3f}'.format(auroc))
        # you can break the loop after 200 epoch
        # if epoch == 199: break

        scheduler.step()

if __name__ == '__main__':
    root_path = "your dataset root path"
    ckpt_path = "your ckpt path"
    setup_seed(111)
    item_list = ['capsule', 'cable','screw','pill','carpet', 'bottle', 'hazelnut','leather', 'grid','transistor', 'metal_nut', 'toothbrush', 'zipper', 'tile', 'wood']
    for i in item_list:
        train(i, root_path, ckpt_path, ifgeom=i in ifgeom)

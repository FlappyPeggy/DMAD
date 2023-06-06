import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.modules.conv as conv

class AddCoords2d(nn.Module):
    def __init__(self):
        super(AddCoords2d, self).__init__()
        self.xx_channel, self.yy_channel = None, None
        self.xx_channel_, self.yy_channel_ = None, None

    def forward(self, input_tensor, feature_size=32):
        batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
        if self.xx_channel is None:
            xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.float32).cuda()
            yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.float32).cuda()
            xx_range = (torch.arange(dim_y, dtype=torch.float32)/(dim_y-1)).cuda()
            yy_range = (torch.arange(dim_x, dtype=torch.float32)/(dim_x-1)).cuda()
            xx_range_ = (torch.arange(feature_size, dtype=torch.float32) / (feature_size - 1)).view(feature_size,1).repeat(1, dim_y//feature_size).flatten().cuda()
            yy_range_ = (torch.arange(feature_size, dtype=torch.float32) / (feature_size - 1)).view(feature_size,1).repeat(1, dim_y//feature_size).flatten().cuda()
            xx_range = xx_range - xx_range_
            yy_range = yy_range - yy_range_

            xx_range = xx_range[None, None, :, None]
            yy_range = yy_range[None, None, :, None]
            xx_channel = torch.matmul(xx_range, xx_ones)
            yy_channel = torch.matmul(yy_range, yy_ones)
            yy_channel = yy_channel.permute(0, 1, 3, 2)
            self.xx_channel = xx_channel * 2 - 1
            self.yy_channel = yy_channel * 2 - 1

            xx_range_ = xx_range_[None, None, :, None]
            yy_range_ = yy_range_[None, None, :, None]
            xx_channel_ = torch.matmul(xx_range_, xx_ones)
            yy_channel_ = torch.matmul(yy_range_, yy_ones)
            yy_channel_ = yy_channel_.permute(0, 1, 3, 2)
            self.xx_channel_ = xx_channel_ * 2 - 1
            self.yy_channel_ = yy_channel_ * 2 - 1
            
        xx_channel = self.xx_channel.repeat(batch_size_shape, 1, 1, 1)
        yy_channel = self.yy_channel.repeat(batch_size_shape, 1, 1, 1)
        xx_channel_ = self.xx_channel_.repeat(batch_size_shape, 1, 1, 1)
        yy_channel_ = self.yy_channel_.repeat(batch_size_shape, 1, 1, 1)
        out = torch.cat([input_tensor, xx_channel, xx_channel_, yy_channel, yy_channel_], dim=1)
        return out

class CoordConv2d(conv.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(CoordConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding)
        self.addcoords2d = AddCoords2d()
        self.conv = nn.Conv2d(in_channels + 4, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, input_tensor):
        out = self.addcoords2d(input_tensor)
        out = self.conv(out)
        return out

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=1, stride=1, padding=0)
        ]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)


class Gradient_Loss(nn.Module):
    def __init__(self, channels=1, alpha=1):
        super(Gradient_Loss, self).__init__()
        self.alpha = alpha
        filter = torch.FloatTensor([[-1., 1.]]).cuda()
        self.filter_x = filter.view(1, 1, 1, 2).repeat(1, channels, 1, 1)
        self.filter_y = filter.view(1, 1, 2, 1).repeat(1, channels, 1, 1)

    def forward(self, gen_frames, gt_frames):
        gen_frames_x = nn.functional.pad(gen_frames, (1, 0, 0, 0))
        gen_frames_y = nn.functional.pad(gen_frames, (0, 0, 1, 0))
        gt_frames_x = nn.functional.pad(gt_frames, (1, 0, 0, 0))
        gt_frames_y = nn.functional.pad(gt_frames, (0, 0, 1, 0))
        gen_dx = nn.functional.conv2d(gen_frames_x, self.filter_x)
        gen_dy = nn.functional.conv2d(gen_frames_y, self.filter_y)
        gt_dx = nn.functional.conv2d(gt_frames_x, self.filter_x)
        gt_dy = nn.functional.conv2d(gt_frames_y, self.filter_y)

        grad_diff_x = torch.abs(gt_dx - gen_dx)
        grad_diff_y = torch.abs(gt_dy - gen_dy)

        return grad_diff_x ** self.alpha + grad_diff_y ** self.alpha


class Smooth_Loss(nn.Module):
    def __init__(self, channels=2, ks=7, alpha=1):
        super(Smooth_Loss, self).__init__()
        self.alpha = alpha
        self.ks = ks
        filter = torch.FloatTensor([[-1 / (ks**2 - 1)] * ks]*ks).cuda()
        filter[ks//2, ks // 2] = 1

        self.filter = filter.view(1, 1, ks, ks).repeat(1, channels, 1, 1)

    def forward(self, gen_frames, weit=None):
        gen_frames = nn.functional.pad(gen_frames, (self.ks // 2, self.ks // 2, self.ks // 2, self.ks // 2))
        smooth = nn.functional.conv2d(gen_frames, self.filter).abs()
        
        if weit is None:
            return (smooth ** self.alpha).mean()
        else:
            return (smooth*weit).sum() / weit.sum()

class Test_Loss(nn.Module):
    def __init__(self, channels=1, ks=(16, 8), alpha=1):
        super(Test_Loss, self).__init__()
        self.alpha = alpha
        self.ks = ks
        self.c = channels
        self.filter = torch.ones((1,1,ks[0], ks[1]),dtype=torch.float32).cuda().repeat(1, channels, 1, 1)/(ks[0]*ks[1])

    def forward(self, gen_frames):
        shape = gen_frames.size()
        b,w,h = shape[0], shape[-2], shape[-1]
        gen_frames = nn.functional.pad(gen_frames.abs().view(b,self.c, w,h), (self.ks[1], self.ks[1], self.ks[0], self.ks[0]))
        gen_dx = nn.functional.conv2d(gen_frames, self.filter).max()
        
        return gen_dx

                                                                        
class VectorQuantizer(nn.Module):
    def __init__(self,
                 num_embeddings=50,
                 embedding_dim=256,
                 beta=0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        self.get_err = Test_Loss(ks=(2,3)) 

    def forward(self, latents, test=False):
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # Get the encoding that has the min distance
        _, idx = dist.topk(2, dim=1, largest=False, sorted=True)
        encoding_inds = idx[:, 0].unsqueeze(1)

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]
        
        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(0, 3, 1, 2).contiguous(), commitment_loss * self.beta ,embedding_loss  # [B x D x H x W]


class Encoder(torch.nn.Module):
    def __init__(self, t_length = 5, n_channel =3, dim=256, r=16):
        super(Encoder, self).__init__()
        
        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=True)
            )
        
        def Basic_(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
            )
        
        self.moduleConv1 = Basic(n_channel*(t_length-1)+3, dim//8)
        self.modulePool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(dim//8, dim//4)
        self.modulePool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.moduleConv3 = Basic(dim//4, dim//2)
        self.modulePool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic_(dim//2, dim)

        self.skip2 = Basic(dim//2, dim//r) 
        self.skip3 = Basic(dim, dim//r)
        
    def forward(self, x):
        tensorConv1 = self.moduleConv1(x)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorConv4 = self.moduleConv4(tensorPool3)
        
        return tensorConv4, self.skip2(tensorConv3.detach()),self.skip3(tensorConv4.detach())


class Decoder(torch.nn.Module):
    def __init__(self, t_length=5, n_channel=3, dim=256, r=16):
        super(Decoder, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                CoordConv2d(intInput, intOutput, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=True),
                CoordConv2d(intOutput, intOutput, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=True)
            )

        def Gen(intInput, intOutput, nc):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )

        def Upsample(nc, intOutput):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=2, padding=1,
                                         output_padding=1, bias=False),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=True)
            )

        def Mask(intInput, intOutput, nc):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.Sigmoid()
            )

        self.moduleConv = Basic(dim+dim//r, dim)
        self.moduleUpsample4 = Upsample(dim, dim//2)

        self.moduleDeconv3 = Basic(dim//2+dim//r, dim//2)
        self.moduleUpsample3 = Upsample(dim//2, dim//4)

        self.moduleDeconv2 = Basic(dim//4,dim//4)
        self.moduleUpsample2 = Upsample(dim//4, dim//8)

        self.moduleDeconv1 = Gen(dim//8, n_channel, dim //8)
        self.mask = Mask(dim//8, 1, dim//8)

    def forward(self, x, skip3, skip4):
        tensorConv = self.moduleConv(torch.cat([x, skip4],dim=1))
        tensorUpsample4 = self.moduleUpsample4(tensorConv)

        tensorDeconv3 = self.moduleDeconv3(torch.cat([tensorUpsample4, skip3],dim=1))
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)

        tensorDeconv2 = self.moduleDeconv2(tensorUpsample3)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)

        output = self.moduleDeconv1(tensorUpsample2)
        mask = self.mask(tensorUpsample2)

        return output, mask

class OffsetNet(torch.nn.Module):
    def __init__(self, t_length=5, n_channel=3, size=None, bn=True):
        super(OffsetNet, self).__init__()
        self.conv_offset1 = nn.Sequential(
            CoordConv2d(n_channel*(t_length-1)+3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            CoordConv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv_offset2 = nn.Sequential(
            ResBlock(64, 64, bn=bn),
            nn.BatchNorm2d(64),
        )
        self.conv_offset3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )
        self.conv_offset4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )
        self.size = size

    def forward(self, x, max_offset1=2, max_offset2=3):
        bs, c, w, h = x.size()

        x1 = self.conv_offset1(x)
        x2 = torch.cat([self.conv_offset2(x1),x1], dim=1)
        offset1 = self.conv_offset4(x2)*2*max_offset1/w
        offset2 = self.conv_offset3(x2)*2*max_offset2/w

        offset1 = F.interpolate(offset1, (w, h), mode='bilinear', align_corners=True).permute(0,2,3,1)
        offset2 = F.interpolate(offset2, (w, h), mode='bilinear', align_corners=True).permute(0,2,3,1)
        
        gridY = torch.linspace(-1, 1, steps=h).view(1, -1, 1, 1).expand(bs, w, h, 1)
        gridX = torch.linspace(-1, 1, steps=w).view(1, 1, -1, 1).expand(bs, w, h, 1)
        grid = torch.cat((gridX, gridY), dim=3).type(torch.float32).cuda()
        grid1 = torch.clamp(offset1 + grid, min=-1, max=1)
        grid2 = torch.clamp(offset2 + grid, min=-1, max=1)
        return grid1, grid2, offset1, offset2


class convAE(torch.nn.Module):
    def __init__(self, n_channel=3,  t_length=5, memory_size=50, memory_dim=256, beta1=1, beta2=0.25, beta3=1, gamma1=1, gamma2=0.25, bkg=None, r=32):
        super(convAE, self).__init__()

        self.encoder = Encoder(t_length, 3, memory_dim, r)
        self.offset_net = OffsetNet(t_length, 3)
        self.decoder = Decoder(t_length, 3,memory_dim, r)

        if bkg is None:
            self.bkg = nn.Parameter(torch.from_numpy(np.zeros((437, 3, 256, 256), dtype=np.float32)))
        else:
            self.bkg = nn.Parameter(torch.from_numpy(bkg.transpose((0, 3, 1, 2)).astype(np.float32) / 127.5 - 1))

        self.vq_layer = VectorQuantizer(memory_size, memory_dim, beta2)
        self.beta = [beta1, beta2, beta3]
        self.gamma = [gamma1, gamma2]
        self.mse1 = 0
        self.mse2 = 0
        self.vq_loss = 0
        self.commit_loss = 0
        self.grad_loss = 0
        self.smooth_loss = 0
        self.offset_loss1 = 0
        self.offset_loss2 = 0
        self.err1 = 0
        self.err2 = 0
        self.err1_ = 0
        self.err2_ = 0
        self.mask_loss = 0

        self.loss_grad = Gradient_Loss(3)
        self.loss_smooth2 = Smooth_Loss(ks=3)
        self.loss_smooth1 = Smooth_Loss(ks=7)
        self.get_err = Test_Loss(ks=(6,18))
        self.get_err_ = Test_Loss(3, ks=(42, 31))

    def forward(self, x, idx, test=False):
        x = torch.cat([x,self.bkg[idx].detach()],dim=1)
        grid1, grid2, offset1, offset2 = self.offset_net(x)

        z_e, skip2, skip3 = self.encoder(x)
        
        quantized_inputs, commit_loss, vq_loss = self.vq_layer(z_e, test)

        z_q, mask = self.decoder(quantized_inputs,skip2, skip3)
        
        z_q = z_q * mask

        z_q_ = F.grid_sample(z_q, grid1, align_corners=True)
        z_q_t = F.grid_sample(z_q_, grid2, align_corners=True)
        mask1 = F.grid_sample(mask, grid1, align_corners=True)
        mask2 = F.grid_sample(mask1, grid2, align_corners=True)

        z_q_ = z_q_ + (1 - mask1) * self.bkg[idx].detach()
        z_q_t = z_q_t + (1 - mask2) * self.bkg[idx].detach()

        return z_q_t, z_q_, z_q, z_e, commit_loss, vq_loss, offset1, offset2, mask, mask1, mask2

    def loss_function(self, x, idx, recon_x, z_q_, z_q, z_e, commit_loss, vq_loss, offset1=None, offset2=None, mask=None, mask1=None, mask2=None, compute_err=False):
        weit = 1+100*torch.abs(F.avg_pool2d((x-self.bkg[idx].detach()).abs()**0.5, kernel_size=15,stride=1, padding=7))    
        self.vq_loss = vq_loss
        self.commit_loss = commit_loss

        if compute_err:
            self.mse2_ = F.mse_loss(recon_x, x, reduction='none')
            self.err1 = self.get_err((offset1 ** 2).sum(-1)**0.5)
            self.err2_ = self.get_err_(self.mse2_)
            return self.mse2_.mean(1)
        else:
            self.mse2 = F.mse_loss(recon_x, x, reduction='none')
            self.mse1 = F.mse_loss(z_q_, x, reduction='none')
            self.mse1 = (self.mse1 * weit).sum() / weit.sum()
            self.mse2 = (self.mse2 * weit).sum() / weit.sum()
            self.grad_loss = self.beta[0]*((self.loss_grad(x, recon_x)*weit).sum()/weit.sum() + (self.loss_grad(x, z_q_)*weit).sum()/weit.sum())

            self.offset_loss1 = (((offset1 ** 2).sum(dim=-1) ** 0.5)*weit[:,0]).sum()/weit[:,0].sum()
            self.offset_loss2 = (((offset2 ** 2).sum(dim=-1) ** 0.5)*weit[:,0]).sum()/weit[:,0].sum()
            self.smooth_loss = self.beta[2] * (self.loss_smooth1(offset1.permute(0, 3, 1, 2), weit) + self.loss_smooth2(offset2.permute(0, 3, 1, 2), weit))
            
            self.mask_loss = -(mask*torch.log(mask+1e-8)).mean() + F.relu(mask.mean()-0.01)*0.1
            self.mask_loss = self.mask_loss * 0.1 + ((1 - mask2.detach())*F.mse_loss(self.bkg[idx],x, reduction='none')).sum() / (1 - mask2.detach()).sum()


            return  self.mse1+self.mse2+self.grad_loss + \
                    self.gamma[0]*(self.vq_loss + self.commit_loss) + \
                    self.gamma[1]*(self.offset_loss1+self.offset_loss2 + self.smooth_loss) +\
                    self.mask_loss

    def load_bkg(self, bkg):
        self.bkg = nn.Parameter(torch.from_numpy(bkg.transpose((0, 3, 1, 2)).astype(np.float32) / 127.5 - 1))

    def latest_losses(self):
        return { 'mse1': self.mse1, 'mse2': self.mse2, 'vq': self.vq_loss, 'commitment': self.commit_loss, 'mask':self.mask_loss, 'offset1':self.offset_loss1, 'offset2':self.offset_loss2, 'err1_':self.err1_, 'err2_':self.err2_, 'err1':self.err1, 'err2':self.err2, 'grad':self.grad_loss, 'smooth':self.smooth_loss}

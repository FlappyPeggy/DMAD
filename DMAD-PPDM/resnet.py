import torch
from torch import Tensor
import torch.nn.modules.conv as conv
import torch.nn.functional as F
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional


__all__ = ['ResNet',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


class AddCoords2d(nn.Module):
    def __init__(self):
        super(AddCoords2d, self).__init__()
        self.xx_channel, self.yy_channel = None, None
        self.xx_channel_, self.yy_channel_ = None, None

    def forward(self, input_tensor, pe_reduction=4):
        batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
        if self.xx_channel is None:
            feature_size = dim_y // pe_reduction
            xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.float32).cuda()
            yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.float32).cuda()
            xx_range = (torch.arange(dim_y, dtype=torch.float32) / (dim_y - 1)).cuda()
            yy_range = (torch.arange(dim_x, dtype=torch.float32) / (dim_x - 1)).cuda()
            xx_range_ = (torch.arange(feature_size, dtype=torch.float32) / (feature_size - 1)).view(feature_size,
                                                                                                            1).repeat(1, dim_y // feature_size).flatten().cuda()
            yy_range_ = (torch.arange(feature_size, dtype=torch.float32) / (feature_size - 1)).view(feature_size,
                                                                                                            1).repeat(1, dim_y // feature_size).flatten().cuda()
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
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, r=4):
        super(CoordConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding)
        self.r = r
        self.addcoords2d = AddCoords2d()
        self.conv = nn.Conv2d(in_channels + 4, out_channels, kernel_size, stride=stride, padding=padding,
                                      bias=bias)

    def forward(self, input_tensor):
        out = self.addcoords2d(input_tensor, self.r)
        out = self.conv(out)
        return out


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        a = self.layer1(x)
        b = self.layer2(a)
        c = self.layer3(b)
        #feature_d = self.layer4(feature_c)

        return [a, b, c]

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        if pretrained:
            try:
                state_dict = torch.load("/data/liuwenrui/modelzoo/wide_resnet50_2-95faca4d.pth")
            except:
                state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        #for k,v in list(state_dict.items()):
        #    if 'layer4' in k or 'fc' in k:
        #        state_dict.pop(k)
        model.load_state_dict(state_dict)
    return model

class AttnBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        attention: bool = True,
    ) -> None:
        super(AttnBasicBlock, self).__init__()
        self.attention = attention
        #print("Attention:", self.attention)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        #self.cbam = GLEAM(planes, 16)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        #if self.attention:
        #    x = self.cbam(x)
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)


        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class AttnBottleneck(nn.Module):
    
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        attention: bool = True,
    ) -> None:
        super(AttnBottleneck, self).__init__()
        self.attention = attention
        #print("Attention:",self.attention)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        #self.cbam = GLEAM([int(planes * self.expansion/4),
        #                   int(planes * self.expansion//2),
        #                   planes * self.expansion], 16)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        #if self.attention:
        #    x = self.cbam(x)
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)


        out += identity
        out = self.relu(out)

        return out

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
    def forward(self, latents):
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]
        
        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight ** 2, dim=1) - \
                2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]
        # dist = torch.matmul(F.normalize(flat_latents, dim=1), F.normalize(self.embedding.weight).t())
        
        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]
        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]
        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]
        
        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents,latents.detach())
        
        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()
        
        return quantized_latents.permute(0, 3, 1,2).contiguous(), commitment_loss * self.beta + embedding_loss  # [B x D x H x W]

class BN_layer(nn.Module):
    def __init__(self,
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers: int,
                 vq: bool,
                 groups: int = 1,
                 width_per_group: int = 64,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 ):
        super(BN_layer, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.groups = groups
        self.base_width = width_per_group
        self.inplanes = 256 * block.expansion
        self.dilation = 1
        self.vq = vq
        self.catwidth = 1024+(16+16+16) * block.expansion if self.vq else 1024*3
        self.bn_layer = self._make_layer(block, 512, layers, stride=2)
        
        if self.vq:
            print("You are running a VQ Version")
            self.conv1_r = conv3x3(64 * block.expansion, 16 * block.expansion)
            self.bn1_r = norm_layer(16 * block.expansion)
            self.conv2_r = conv3x3(128 * block.expansion, 16 * block.expansion)
            self.bn2_r = norm_layer(16 * block.expansion)
            self.conv3_r = conv3x3(256 * block.expansion, 16 * block.expansion)
            self.bn3_r = norm_layer(16 * block.expansion)

            self.conv1 = conv3x3((16+0) * block.expansion, 16 * block.expansion, 2)
            self.bn1 = norm_layer(16 * block.expansion)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(16 * block.expansion, 16 * block.expansion, 2)
            self.bn2 = norm_layer(16* block.expansion)
            self.conv3 = conv3x3((0+16) * block.expansion,16 * block.expansion, 2)
            self.bn3 = norm_layer(16 * block.expansion)
            self.conv4 = conv1x1((256) * block.expansion, 256 * block.expansion, 1)
            self.bn4 = norm_layer(256 * block.expansion)

            #self.vq1 = VectorQuantizer(400, 256)
            #self.vq2 = VectorQuantizer(400, 512)
            self.vq3 = VectorQuantizer(200, 1024)
        else:
            self.conv1 = conv3x3((64) * block.expansion, 128 * block.expansion, 2)
            self.bn1 = norm_layer(128 * block.expansion)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(128 * block.expansion, 256 * block.expansion, 2)
            self.bn2 = norm_layer(256* block.expansion)
            self.conv3 = conv3x3((128) * block.expansion, 256 * block.expansion, 2)
            self.bn3 = norm_layer(256 * block.expansion)
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.catwidth, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.catwidth, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        l1, l2, l3 = x[0], x[1], x[2]
        if self.vq:
            loss1, loss2 = 0, 0
            l3_, loss3 = self.vq3(l3)
            
            l1 = self.relu(self.bn1_r(self.conv1_r(l1)))
            l2 = self.relu(self.bn2_r(self.conv2_r(l2)))
            l3 = self.relu(self.bn3_r(self.conv3_r(l3)))
            
            l1_ = self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(l1))))))
            l2_ = self.relu(self.bn3(self.conv3(l2)))
            
            feature = torch.cat([l1_,l2_,l3, l3_],1)
        else:
            l1 = self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(l1))))))
            l2 = self.relu(self.bn3(self.conv3(l2)))
            feature = torch.cat([l1,l2,l3],1)
        output = self.bn_layer(feature)
    
        if self.vq:
            return output.contiguous(), loss1+loss2+loss3
        else:
            return output.contiguous(), 0

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class OffsetNet(torch.nn.Module):
    def __init__(self, gamma=1, n_channel=3, bn=True):
        super(OffsetNet, self).__init__()

        self.gamma = gamma

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

        self.conv_offset1 = nn.Sequential(
            CoordConv2d(n_channel, 64, kernel_size=3, stride=1, padding=1),
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
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
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
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )
        self.conv_offset3_ = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),                                                     
            )
        self.conv_offset4_ = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            )


        class Smooth_Loss(nn.Module):
            def __init__(self, channels=2, ks=7, alpha=1):
                super(Smooth_Loss, self).__init__()
                self.alpha = alpha
                self.ks = ks
                filter = torch.FloatTensor([[-1 / (ks ** 2 - 1)] * ks] * ks).cuda()
                filter[ks // 2, ks // 2] = 1

                self.filter = filter.view(1, 1, ks, ks).repeat(1, channels, 1, 1)

            def forward(self, gen_frames):
                gen_frames = nn.functional.pad(gen_frames, (self.ks // 2, self.ks // 2, self.ks // 2, self.ks // 2))
                smooth = nn.functional.conv2d(gen_frames, self.filter).abs()

                return (smooth ** self.alpha).mean()

        self.loss_smooth2 = Smooth_Loss(ks=3)
        self.loss_smooth1 = Smooth_Loss(ks=3)
        self.scale1 = 2 * 8 / 256
        self.scale2 = 2 * 3 / 256


    def forward(self, x, test=False):
        bs, c, w, h = x.size()

        x1 = self.conv_offset1(x)
        x2 = torch.cat([self.conv_offset2(x1), x1], dim=1)
        offset1 = self.conv_offset4(x2)*self.scale1
        offset2 = self.conv_offset3(x2)*self.scale2
        offset1_ = self.conv_offset4_(x2)*self.scale1
        offset2_ = self.conv_offset3_(x2)*self.scale2

        offset1 = F.interpolate(offset1, (w, h), mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
        offset2 = F.interpolate(offset2, (w, h), mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
        offset1_ = F.interpolate(offset1_, (w, h), mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
        offset2_ = F.interpolate(offset2_, (w, h), mode='bilinear', align_corners=True).permute(0, 2, 3, 1)

        gridY = torch.linspace(-1, 1, steps=h).view(1, -1, 1, 1).expand(bs, w, h, 1)
        gridX = torch.linspace(-1, 1, steps=w).view(1, 1, -1, 1).expand(bs, w, h, 1)
        grid = torch.cat((gridX, gridY), dim=3).type(torch.float32).cuda()
        grid1 = torch.clamp(offset1 + grid, min=-1, max=1)
        grid2 = torch.clamp(offset2 + grid, min=-1, max=1)
        grid1_ = torch.clamp(offset1_ + grid, min=-1, max=1)
        grid2_ = torch.clamp(offset2_ + grid, min=-1, max=1)

        out1 = F.grid_sample(x, grid1, align_corners=True)
        out2 = F.grid_sample(out1, grid2, align_corners=True)

        if test:
            return out2, (offset1 ** 2).sum(dim=-1) ** 0.5, (offset2 ** 2).sum(dim=-1) ** 0.5, (offset1_ ** 2).sum(dim=-1) ** 0.5, (offset2_ ** 2).sum(dim=-1) ** 0.5, grid1_, grid2_, grid2
        else:
            transposed = F.grid_sample(F.grid_sample(out2, grid2_, align_corners=True), grid1_, align_corners=True)

            offset_loss = ((offset1 ** 2).sum(dim=-1) ** 0.5).mean()+((offset2 ** 2).sum(dim=-1) ** 0.5).mean() + \
            self.loss_smooth1(offset1.permute(0, 3, 1, 2))+self.loss_smooth2(offset2.permute(0, 3, 1, 2)) + \
            self.loss_smooth1(offset1_.permute(0, 3, 1, 2)) + self.loss_smooth2(offset2_.permute(0, 3, 1, 2)) + \
                        F.mse_loss(transposed, x) * self.gamma

            return out1, out2, offset_loss

def wide_resnet50_2(pretrained: bool = False, progress: bool = True, vq: bool = False, gamma: float = 1., **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs), BN_layer(AttnBottleneck,3,vq, **kwargs), OffsetNet(gamma)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, vq: bool = False, gamma: float = 1., **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs), BN_layer(AttnBottleneck,3,vq,**kwargs),  OffsetNet(gamma)



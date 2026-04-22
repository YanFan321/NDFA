'''
resnet for cifar in pytorch
Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
'''
import logging

import torch
import torch.nn as nn
import math

from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from collections import OrderedDict
from utils.OutputTrainingHook import OutputTrainingHook, GetOutputTrainingHook
from .configs import network_config
from .augdfa_nets import AlignmentProbe, AuxDFANetwork


def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout_rate=0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class LocalResNet(nn.Module):

    def __init__(self, block, layers, arch, local_module_num, class_num=10, dfa_num=16,
                 wide_list=(16, 16, 32, 64), dropout_rate=0, hidden_dim=128,train_mode='NDFA'
                 ):
        super(LocalResNet, self).__init__()

        assert arch in ['resnet32', 'resnet110'], "This repo supports resnet32 and resnet110 currently. " \
                                                  "For other networks, please set network configs in .configs."

        self.inplanes = wide_list[0]
        self.dropout_rate = dropout_rate
        self.feature_num = wide_list[-1]
        self.class_num = class_num
        self.dfa_num = dfa_num
        self.local_module_num = local_module_num
        self.train_mode = train_mode
        self.layers = [1] + layers
        self.conv1 = nn.Conv2d(3, wide_list[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(wide_list[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, wide_list[1], layers[0])
        self.layer2 = self._make_layer(block, wide_list[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, wide_list[3], layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.feature_num, self.class_num)
        self.output_hook = OutputTrainingHook()

        self.criterion_ce = nn.CrossEntropyLoss()
        if local_module_num == 1:
            self.net_config = []
        else:
            self.net_config = network_config[arch]['net_config'][:local_module_num - 1]

            logging.info(f"net_config: {self.net_config}")
        self.net_dim = network_config[arch]['net_dim']

        for j, item in enumerate(self.net_config):
            module_index, layer_index, layer_num = item

            exec('self.aug_dfa_' + str(module_index) + '_' + str(layer_index) +
                 '= AuxDFANetwork('
                 'class_num=class_num, '
                 'hidden_dim=hidden_dim, '
                 'train_mode=train_mode)')

        # for module_index in (0, 1, 2, 3):
        #     for layer_index in range(self.layers[module_index]):
        #         exec('self.aug_dfa_' + str(module_index) + '_' + str(layer_index) +
        #              '= AuxDFANetwork('
        #              'class_num=class_num, '
        #              'hidden_dim=hidden_dim, '
        #              'train_mode=train_mode)')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, dropout_rate=self.dropout_rate))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dropout_rate=self.dropout_rate))

        return nn.Sequential(*layers)

    def _local_forward(self, x, target, stage_i, layer_i, grad_at_output):
        
        loss_ixy, hidden_logits = eval('self.aug_dfa_' + str(stage_i) + '_' + str(layer_i))(x, target, grad_at_output)
        return hidden_logits, loss_ixy

    def forward(self, img, target):
        if self.train_mode == 'BP':
            for stage_i in (0, 1, 2, 3):
                for layer_i in range(self.layers[stage_i]):
                    if stage_i == 0:
                        assert layer_i == 0
                        x = self.conv1(img)
                        x = self.bn1(x)
                        x = self.relu(x)
                    else:
                        x = eval('self.layer' + str(stage_i))[layer_i](x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc1_dfa(x)
            x = self.fc(x)
            loss = self.criterion_ce(x, target)
            if self.training:
                loss.backward()
            return x, loss
        else:
            grad_at_output = torch.zeros([img.shape[0], self.class_num], requires_grad=False).to(img.device)

            aug_loss = 0
            for stage_i in (0, 1, 2, 3):
                for layer_i in range(self.layers[stage_i]):
                    if stage_i == 0:
                        assert layer_i == 0
                        x = self.conv1(img)
                        x = self.bn1(x)
                        x = self.relu(x)
                    else:
                        x = eval('self.layer' + str(stage_i))[layer_i](x)
                    if self.dfa_num == 16:
                        if (stage_i ==3 and layer_i ==4):
                            continue
                        else:
                            hidden_logits, loss_ixy_dfa = self._local_forward(x,
                                                                              target,
                                                                              stage_i,
                                                                              layer_i,
                                                                              grad_at_output)
                            if self.training:
                                aug_loss = aug_loss + loss_ixy_dfa
                                x = x.detach()
                if self.dfa_num == 4:
                    if stage_i != 3:
                        hidden_logits, loss_ixy_dfa = self._local_forward(x,
                                                                          target,
                                                                          stage_i,
                                                                          0,
                                                                          grad_at_output)
                        if self.training:
                            aug_loss = aug_loss + loss_ixy_dfa
                            x = x.detach()
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            x = self.output_hook(x, grad_at_output)
            loss = self.criterion_ce(x, target)
            if self.training:
                loss = loss + aug_loss
                loss.backward()
            return x, loss

def resnet32(**kwargs):
    model = LocalResNet(BasicBlock, [5, 5, 5], arch='resnet32', **kwargs)
    return model


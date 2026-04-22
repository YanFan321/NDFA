import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .augdfa_nets import AlignmentProbe, AuxDFANetwork
from utils.OutputTrainingHook import OutputTrainingHook


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout_rate=0.0):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate and dropout_rate > 0 else nn.Identity()
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

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

        out = out + residual
        out = self.relu(out)
        return out

class LocalResNetImageNet32(nn.Module):

    def __init__(
        self,
        train_mode: str,
        hidden_dim: int = 128,
        class_num: int = 1000,
        layers=(3, 4, 4, 4),
        wide_list=(64, 64, 128, 256, 512),
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        assert len(layers) == 4, "layers must have 4 stage depths, e.g. (3,4,4,4) for ResNet32."
        assert len(wide_list) == 5, "wide_list must be (stem, s1, s2, s3, s4)."

        self.train_mode = train_mode
        self.class_num = class_num
        self.dropout_rate = dropout_rate

        self.inplanes = wide_list[0]
        self.conv1 = nn.Conv2d(3, wide_list[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(wide_list[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, wide_list[1], layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, wide_list[2], layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, wide_list[3], layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, wide_list[4], layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(wide_list[4] * BasicBlock.expansion, class_num)

        self.output_hook = OutputTrainingHook()

        self.layers = [1] + list(layers)

        for stage_i in range(0, 5):  # 0..4
                exec(f"self.layer_dfa_{stage_i}_0 = "
                     f"AuxDFANetwork(class_num=class_num, hidden_dim=hidden_dim, train_mode=train_mode)"
                )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    m.weight.data.fill_(1)
                if hasattr(m, "bias") and m.bias is not None:
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dropout_rate=self.dropout_rate))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, downsample=None, dropout_rate=self.dropout_rate))
        return nn.Sequential(*layers)

    def _stem(self, img):
        x = self.conv1(img)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

    def forward(self, img, target):

        grad_at_output = torch.zeros([img.shape[0], self.class_num], requires_grad=False).to(img.device)

        # stage 0 (stem)
        x = self._stem(img)
        DFA_loss = 0

        # stages 1..4
        for stage_i, layer_mod in enumerate([self.layer1, self.layer2, self.layer3, self.layer4], start=1):

            x = layer_mod(x)
            if stage_i == 4 or stage_i == 2:
                dfa_loss, _ = getattr(self, f"layer_dfa_{stage_i}_0")(x, target, grad_at_output)
                if self.training:
                    DFA_loss = DFA_loss + dfa_loss
                    x = x.detach()

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.output_hook(x, grad_at_output)
        loss = F.cross_entropy(x, target)
        if self.training:
            loss = loss + DFA_loss
            loss.backward()
        return x, loss


def resnet32_imagenet(**kwargs) -> LocalResNetImageNet32:

    return LocalResNetImageNet32(layers=(3, 4, 4, 4), **kwargs)

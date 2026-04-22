import torch
import torch.nn as nn
from utils.OutputTrainingHook import OutputTrainingHook
from .augdfa_nets import AlignmentProbe, AuxDFANetwork
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self, train_mode, dim, class_num):
        super(AlexNet, self).__init__()
        self.train_mode = train_mode
        self.dim = dim
        self.class_num = class_num

        self.features = conv_features(class_num, dim, train_mode)

        self.classifier = linear_classifier(class_num, dim, train_mode)

        self.output_hook = OutputTrainingHook()

    def forward(self, x, target):
        if self.train_mode == 'BP':
            x = self.features(x, target, None)
            x = self.classifier(x, target, None)
            loss = F.cross_entropy(x, target)
            if self.training:
                loss.backward()
            return x, loss

        grad_at_output = nn.Parameter(torch.zeros([x.shape[0], self.class_num]), requires_grad=False).to(x.device)

        x, f_loss = self.features(x, target, grad_at_output)
        x, c_loss = self.classifier(x, target, grad_at_output)

        x = self.output_hook(x, grad_at_output)
        loss = F.cross_entropy(x, target)
        if self.training:
            loss = loss + f_loss + c_loss
            loss.backward()
        return x, loss

class conv_features(nn.Module):
    def __init__(self, class_num, dim, train_mode):
        super(conv_features, self).__init__()
        self.train_mode = train_mode

        self.conv_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn_1 = nn.BatchNorm2d(64)
        self.conv_1_dfa = AuxDFANetwork(class_num, dim, train_mode,64*16*16)

        self.conv_2 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn_2 = nn.BatchNorm2d(192)
        self.conv_2_dfa = AuxDFANetwork(class_num, dim, train_mode,192*8*8)

        self.conv_3 = nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1)
        self.bn_3 = nn.BatchNorm2d(384)
        self.conv_3_dfa = AuxDFANetwork(class_num, dim, train_mode,384*8*8)

        self.conv_4 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.bn_4 = nn.BatchNorm2d(256)
        self.conv_4_dfa = AuxDFANetwork(class_num, dim, train_mode,256*8*8)

        self.conv_5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn_5 = nn.BatchNorm2d(256)
        self.pool_5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_5_dfa = AuxDFANetwork(class_num, dim, train_mode,4096)

        # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.drop = nn.Dropout()

        self.act = nn.ReLU(inplace=True)

    def forward(self, x, target, grad_at_output):
        if self.train_mode == 'BP':
            x = self.conv_1(x)
            x = self.bn_1(x)
            x = self.act(x)
            x = self.pool_1(x)

            x = self.conv_2(x)
            x = self.bn_2(x)
            x = self.act(x)
            x = self.pool_2(x)

            x = self.conv_3(x)
            x = self.bn_3(x)
            x = self.act(x)

            x = self.conv_4(x)
            x = self.bn_4(x)
            x = self.act(x)

            x = self.conv_5(x)
            x = self.bn_5(x)
            x = self.act(x)
            x = self.pool_5(x)
            x = torch.flatten(x, 1)
            return x

        aug_loss = 0
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.act(x)
        x = self.pool_1(x)
        loss1, hidden_logits1 = self.conv_1_dfa(x, target, grad_at_output)
        if self.training:
            aug_loss = aug_loss + loss1
            x = x.detach()
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.act(x)
        x = self.pool_2(x)
        loss2, hidden_logits2 = self.conv_2_dfa(x, target, grad_at_output)
        if self.training:
            aug_loss = aug_loss + loss2
            x = x.detach()
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.act(x)
        loss3, hidden_logits3 = self.conv_3_dfa(x, target, grad_at_output)
        if self.training:
            aug_loss = aug_loss + loss3
            x = x.detach()
        x = self.conv_4(x)
        x = self.bn_4(x)
        x = self.act(x)
        loss4, hidden_logits4 = self.conv_4_dfa(x, target, grad_at_output)
        if self.training:
            aug_loss = aug_loss + loss4
            x = x.detach()
        x = self.conv_5(x)
        x = self.bn_5(x)
        x = self.act(x)
        x = self.pool_5(x)

        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        #x = self.drop(x)

        loss5, hidden_logits5 = self.conv_5_dfa(x, target, grad_at_output)
        if self.training:
            aug_loss = aug_loss + loss5
            x = x.detach()
        return x, aug_loss

class linear_classifier(nn.Module):
    def __init__(self, class_num, dim, train_mode):
        super(linear_classifier, self).__init__()
        self.train_mode = train_mode
        self.drop = nn.Dropout()
        self.act = nn.ReLU(inplace=True)

        self.linear_1 = nn.Linear(256 * 4 * 4, 4096)
        self.linear_1_dfa = AuxDFANetwork(class_num, dim, train_mode,4096)

        self.linear_2 = nn.Linear(4096, 4096)
        # self.fc1_dfa = AuxDFANetwork(class_num, dim, train_mode,4096)
        self.fc1_dfa = AlignmentProbe()

        self.linear_3 = nn.Linear(4096, class_num)

    def forward(self, x, target, grad_at_output):
        if self.train_mode == 'BP':
            x = self.linear_1(x)
            x = self.act(x)
            x = self.linear_2(x)
            x = self.act(x)
            x = self.linear_3(x)
            return x

        aug_loss = 0
        x = self.linear_1(x)
        x = self.act(x)
        #x = self.drop(x)
        loss1, hidden_logits1 = self.linear_1_dfa(x, target, grad_at_output)
        if self.training:
            aug_loss = aug_loss + loss1
            x = x.detach()
        x = self.linear_2(x)
        x = self.act(x)
        x = self.fc1_dfa(x)

        x = self.linear_3(x)

        return x, aug_loss

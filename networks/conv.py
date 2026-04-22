import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.OutputTrainingHook import OutputTrainingHook
from .augdfa_nets import AlignmentProbe, AuxDFANetwork



class ConvNet2FC2(nn.Module):
    def __init__(self, train_mode, dim, class_num):
        super(ConvNet2FC2, self).__init__()
        self.train_mode = train_mode
        self.dim = dim
        self.class_num = class_num

        self.conv1 = nn.Conv2d(3, 15, 5, 1)
        self.conv1_bn = nn.BatchNorm2d(15)
        self.conv1_dfa = AuxDFANetwork(class_num, dim, train_mode)

        self.conv2 = nn.Conv2d(15, 40, 5, 1)
        self.conv2_bn = nn.BatchNorm2d(40)
        self.conv2_dfa = AuxDFANetwork(class_num, dim, train_mode)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(5760, 128)
        self.fc1_bn = nn.BatchNorm1d(128)

        self.fc_out = nn.Linear(128, class_num)

        self.output_hook = OutputTrainingHook()


    def forward(self, x, target):
        if self.train_mode == 'BP':
            x = self.conv1(x)
            x = self.conv1_bn(x)
            x = F.relu(x)

            x = self.conv2(x)
            x = self.conv2_bn(x)
            x = F.relu(x)

            x = self.pool(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = self.fc1_bn(x)
            x = F.relu(x)

            x = self.fc_out(x)
            loss = F.cross_entropy(x, target)
            if self.training:
                loss.backward()
            return x, loss
        aug_loss = 0
        grad_at_output = torch.zeros([x.shape[0], self.class_num], requires_grad=False).to(x.device)
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        loss1, hidden_logit1 = self.conv1_dfa(x, target, grad_at_output)
        if self.training:
            aug_loss = aug_loss + loss1
            x = x.detach()

        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)
        loss2, hidden_logit2 = self.conv2_dfa(x, target, grad_at_output)
        if self.training:
            aug_loss = aug_loss + loss2
            x = x.detach()

        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.relu(x)

        x = self.fc_out(x)
        x = self.output_hook(x, grad_at_output)
        loss = F.cross_entropy(x, target)
        if self.training:
            loss = loss + aug_loss
            loss.backward()
        return x, loss

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.OutputTrainingHook import GetOutputTrainingHook, OutputTrainingHook


class AlignmentProbe(nn.Module):
    def forward(self, x):
        return x
class AuxDFANetwork(nn.Module):
    def __init__(self, class_num=10, hidden_dim=256, train_mode='NDFA', width=None):
        super(AuxDFANetwork, self).__init__()

        self.criterion = nn.CrossEntropyLoss()
        self.fc_out_channels = class_num
        self.hidden_dim = hidden_dim
        self.Getoutput_hook = GetOutputTrainingHook()
        self.output_hook = OutputTrainingHook()
        self.train_mode = train_mode
        aux_net = []
        self._frozen_once = False
        if width is None:
            aux_net.extend([
                            nn.AdaptiveAvgPool2d((5, 5)),
                            nn.Flatten(),
                            nn.LazyLinear(self.hidden_dim),
                            nn.BatchNorm1d(self.hidden_dim),
                            nn.Sigmoid(),
                            nn.Linear(self.hidden_dim, self.fc_out_channels)])
        else:
            aux_net.extend([
                            nn.Flatten(),
                            nn.Linear(width, self.hidden_dim),
                            nn.BatchNorm1d(self.hidden_dim),
                            nn.Sigmoid(),
                            nn.Linear(self.hidden_dim, self.fc_out_channels)])

        self.head = nn.Sequential(*aux_net)

    def _prepare_head_input(self, x):
        # ViT blocks produce token features [B, N, C]. Mirror the
        # NonLinearB2W expand behavior by operating on the cls token only;
        # autograd then routes gradients back to token 0 in the original tensor.
        if x.dim() == 3:
            return x[:, 0, :].contiguous()
        return x

    def _head_has_pooling(self):
        return len(self.head) > 0 and isinstance(self.head[0], nn.AdaptiveAvgPool2d)

    def _forward_head(self, x, detach_input=False):
        if detach_input:
            x = x.detach()
        x = self._prepare_head_input(x)

        if self._head_has_pooling() and x.dim() == 2:
            return self.head[1:](x)
        return self.head(x)

    def _forward_frozen_head(self, x):
        x = self._prepare_head_input(x)
        modules = list(self.head.children())
        idx = 0
        if self._head_has_pooling():
            if x.dim() != 2:
                x = modules[0](x)
            idx = 1

        x = modules[idx](x)
        idx += 1
        tail = modules[idx:]

        if len(tail) == 1:
            linear = tail[0]
            return F.linear(
                x,
                linear.weight.detach(),
                linear.bias.detach() if linear.bias is not None else None
            )

        if len(tail) == 3:
            linear1, act, linear2 = tail
            x = F.linear(
                x,
                linear1.weight.detach(),
                linear1.bias.detach() if linear1.bias is not None else None
            )
            x = act(x)
            return F.linear(
                x,
                linear2.weight.detach(),
                linear2.bias.detach() if linear2.bias is not None else None
            )

        if len(tail) == 4:
            linear1, bn1, act, linear2 = tail
            x = F.linear(
                x,
                linear1.weight.detach(),
                linear1.bias.detach() if linear1.bias is not None else None
            )
            x = F.batch_norm(
                x,
                bn1.running_mean,
                bn1.running_var,
                weight=bn1.weight.detach() if bn1.weight is not None else None,
                bias=bn1.bias.detach() if bn1.bias is not None else None,
                training=False,
                momentum=bn1.momentum,
                eps=bn1.eps,
            )

            x = act(x)
            return F.linear(
                x,
                linear2.weight.detach(),
                linear2.bias.detach() if linear2.bias is not None else None
            )

        raise ValueError(f"Unsupported aux head layout with {len(modules)} modules.")



    def forward(self, x, target, grad_at_output):
        local_features = self._forward_head(x, detach_input=True)
        loss = self.criterion(local_features, target)
        features = self._forward_frozen_head(x)
        features = self.Getoutput_hook(features, grad_at_output)
        loss_global = self.criterion(features, target)
        loss += loss_global

        return loss, features.detach()

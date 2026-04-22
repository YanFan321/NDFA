import argparse
import logging
import os
import shutil
import time
import math
import numpy as np
from datetime import datetime
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from datasets import load_dataset as hf_load_dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from utils.others import *
import json
import random
import networks.vits
from torchvision.datasets import ImageFolder
import timm
from torchvision.transforms import InterpolationMode
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy

parser = argparse.ArgumentParser(description='Implementation of AugLocal in PyTorch')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset: [cifar10|stl10|svhn|imagenet|imagenette|tiny-imagenet]')
parser.add_argument('--model', default='vits', type=str, help='model: [vit-tiny|vit-small|vits]')
parser.add_argument('--droprate', default=0.0, type=float, help='dropout probability (default: 0.0)')
parser.add_argument('--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='', type=str, help='name of experiment')
parser.add_argument('--cos_lr', dest='cos_lr', action='store_true', default=True)

parser.add_argument('--warmup_epochs', default=10, type=int, help='warmup epochs for cosine lr')
parser.add_argument('--min_lr', default=1e-6, type=float, help='minimum lr for cosine schedule')
parser.add_argument('--randaug', action='store_true', default=True, help='use RandAugment for CIFAR-like datasets')
parser.add_argument('--mixup', type=float, default=0.0, help='mixup alpha, mixup enabled if > 0')
parser.add_argument('--cutmix', type=float, default=0.0, help='cutmix alpha, cutmix enabled if > 0')
parser.add_argument('--cutmix_minmax', type=float, nargs=2, default=None, metavar=('MIN', 'MAX'),
                    help='cutmix min/max ratio, overrides --cutmix if set (e.g., 0.2 1.0)')
parser.add_argument('--mixup_prob', type=float, default=1.0, help='probability of applying mixup or cutmix')
parser.add_argument('--mixup_switch_prob', type=float, default=0.5, help='probability of switching to cutmix when both enabled')
parser.add_argument('--mixup_mode', type=str, default='batch', choices=['batch', 'pair', 'elem'], help='mixup mode')
parser.add_argument('--smoothing', type=float, default=0.0, help='label smoothing (used when mixup/cutmix disabled)')
parser.add_argument('--epochs', default=300, type=int, help='number of training epochs')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--dfa_num', default=12, type=int, help='number of DFA/Aux injections for vits')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--wd', default=5e-2, type=float, help='weight decay')
parser.add_argument('--save_path', default='', type=str, help='the directory used to save the trained models')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--train_mode', default='BP', type=str, help='the train_mode')
parser.add_argument('--vit_drop_path', default=0.1, type=float, help='ViT drop path rate')
parser.add_argument('--depth', default=12, type=int, help='ViT depth')
parser.add_argument('--hidden_dim', default=128, type=int)

parser.add_argument('--vit_weight_matching', action='store_true', default=False, help='Enable DFA weight matching init (optional)')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--device', default=str(device), type=str, help='the device')
args = parser.parse_args()

class HFDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, hf_ds, transform=None):
        self.hf_ds = hf_ds
        self.transform = transform

    def __len__(self):
        return self.hf_ds.num_rows

    def __getitem__(self, idx):
        ex = self.hf_ds[idx]
        img = ex["image"]
        if hasattr(img, "mode") and img.mode != "RGB":
            img = img.convert("RGB")
        label = int(ex["label"])
        if self.transform is not None:
            img = self.transform(img)
        return img, label


if args.save_path == '':
    save_path = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = save_path + args.name + '_' + str(args.seed)
else:
    save_path = args.save_path

if not os.path.exists(save_path):
    os.makedirs(save_path)
# Logging settings
setup_logging(os.path.join(save_path, 'log.txt'))
logging.info('saving to:' + str(save_path))

# Configurations adopted for training deep networks.
training_configurations = {
    'vits': {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'initial_learning_rate': args.lr,
        'changing_lr': [80, 120],
        'lr_decay_rate': 0.1,
        'momentum': 0.9,
        'weight_decay': args.wd,
        'patch_size': 16,
        'in_chans': 3,
        'embed_dim': 192,
        'depth': args.depth,
        'num_heads': 3,
        'mlp_ratio': 4
    },
}

def pad_image(x):
    return F.pad(x.unsqueeze(0), (4, 4, 4, 4), mode='reflect').squeeze(0)

def main():
    set_random_seed(seed=args.seed)
    global best_prec1
    best_prec1 = 0
    global val_acc
    val_acc = []

    if args.dataset == 'imagenet':
        class_num = 1000
    elif args.dataset in ['cifar10']:
        class_num = 10
    elif args.dataset in ['cifar100']:
        class_num = 100

    if args.dataset in ['imagenet']:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        kwargs_dataset_train = {}
        kwargs_dataset_test = {}
    elif 'cifar' in args.dataset:
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        kwargs_dataset_train = {'train': True}
        kwargs_dataset_test = {'train': False}
    else:
        normalize = transforms.Normalize(mean=[x / 255 for x in [127.5, 127.5, 127.5]],
                                         std=[x / 255 for x in [127.5, 127.5, 127.5]])
        kwargs_dataset_train = {'split': 'train'}
        kwargs_dataset_test = {'split': 'test'}

    if args.dataset in ['imagenet']:
        image_size = 224

        ra = transforms.RandAugment(num_ops=2, magnitude=9)

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.08, 1.0),
                ratio=(3. / 4., 4. / 3.),
                interpolation=InterpolationMode.BICUBIC,  # ViT 常用 bicubic
            ),
            transforms.RandomHorizontalFlip(),
            ra,
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        ])
    elif 'cifar' in args.dataset:

        # CIFAR images are 32x32; use a stronger augmentation recipe for training from scratch
        ra = transforms.RandAugment(num_ops=2, magnitude=9) if args.randaug else transforms.Lambda(lambda x: x)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            ra,  # PIL-space augmentation
            transforms.ToTensor(),
            normalize,
            # Mild RandomErasing helps ViT generalize on CIFAR.
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.20), ratio=(0.3, 3.3), value='random'),
        ])
        image_size = 32
    else:
        raise NotImplementedError
    if args.dataset in ['imagenet']:
        transform_test = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
        ])
    elif args.dataset in ['cifar10','cifar100']:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    kwargs = {'num_workers': 6, 'pin_memory': True}

    if args.dataset in ['imagenet']:
        train_dir = os.path.join("../image_datasets", args.dataset, "train")
        val_dir = os.path.join("../image_datasets", args.dataset, "val")
        train_dataset = ImageFolder(train_dir, transform=transform_train)
        val_dataset = ImageFolder(val_dir, transform=transform_test)
    else:
        train_dataset = datasets.__dict__[args.dataset.upper()]('../image_datasets/' + args.dataset, download=True,
                                                            transform=transform_train,
                                                            **kwargs_dataset_train)
        val_dataset = datasets.__dict__[args.dataset.upper()]('../image_datasets/' + args.dataset, transform=transform_test, download=True,
                                                          **kwargs_dataset_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=training_configurations[args.model]['batch_size'], shuffle=True, drop_last=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=training_configurations[args.model]['batch_size'], shuffle=False, **kwargs)

    with open(save_path + '/args.json', 'w') as fid:
        json.dump(args.__dict__, fid, indent=2)

    logging.info('args: ' + str(args))

    # create model
    patch_size = training_configurations[args.model]['patch_size']
    if 'cifar' in args.dataset and patch_size == 16:
        patch_size = 4
    model = networks.vits.ViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=class_num,
        dim=192,
        depth=training_configurations[args.model]['depth'],
        heads=3,
        mlp_dim=192*4,
        train_mode=args.train_mode,
        class_num=class_num,
        B_dim=args.hidden_dim,
        dfa_num=args.dfa_num,
    )

    model.eval()

    logging.info(str(model))

    cudnn.benchmark = True

    model = model.cuda()
    validate(val_loader, model, 0, None)

    # Mixup / CutMix (works for both CIFAR and ImageNet).
    mixup_active = (args.mixup > 0.0) or (args.cutmix > 0.0) or (args.cutmix_minmax is not None)
    mixup_fn = None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=0.0,  # smoothing handled by soft targets; keep 0 here
            num_classes=class_num
        )
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing) if args.smoothing > 0.0 else torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)


    start_epoch = 0
    for epoch in range(start_epoch, training_configurations[args.model]['epochs']):
        adjust_learning_rate(optimizer, epoch + 1)

        # train for one epoch
        train_acc, loss_output = train(train_loader, model, optimizer, criterion, mixup_fn, epoch, None)

        # evaluate on validation set
        prec1 = validate(val_loader, model, epoch, None)

        # train_eval_acc = validate(train_loader, model, epoch, None)
        # print("train (eval mode) acc:", train_eval_acc)
        logging.info('train acc {:.4f} test acc {:.4f}'.format(train_acc, prec1))
        best_prec1 = max(prec1, best_prec1)
        logging.info(args.train_mode + args.dataset + ' Best accuracy: ' + str(best_prec1))
def train(train_loader, model, optimizer, criterion, mixup_fn, epoch, writer):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    train_batches_num = len(train_loader)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (x, target) in enumerate(train_loader):
        if args.device == 'cuda':
            target = target.cuda(non_blocking=True)
            x = x.cuda(non_blocking=True)

        if mixup_fn is not None:
            x, target = mixup_fn(x, target)

        optimizer.zero_grad()

        output, aug_loss = model(x,target)
        loss = criterion(output, target) + aug_loss

        loss.backward()
        optimizer.step()
        # For logging accuracy with mixup/cutmix, convert soft targets back to hard labels.
        hard_target = target.argmax(dim=1) if target.ndim == 2 else target
        prec1 = accuracy(output.data, hard_target, topk=(1,))[0]
        losses.update(loss.data.item(), target.size(0))
        top1.update(prec1.item(), target.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) == len(train_loader):
            # print(discriminate_weights)

            string = ('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                      'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                      'Prec@1 {top1.value:.3f} ({top1.ave:.3f})'.format(
                epoch, i + 1, train_batches_num, batch_time=batch_time,
                loss=losses, top1=top1))
            if args.device == 'cuda':
                string += (
                'mem={:.0f}MiB, max_mem={:.0f}MiB, max_mem={:.0f}MiB \n'.format(torch.cuda.memory_allocated() / 1e6,
                                                                                torch.cuda.max_memory_allocated() / 1e6,
                                                                                torch.cuda.max_memory_reserved() / 1e6))

            logging.info(string)
    return top1.ave, losses.ave


def validate(val_loader, model, epoch, writer):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    train_batches_num = len(val_loader)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if args.device == 'cuda':
            target = target.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)

        with torch.no_grad():
            output, aug_loss = model(input, target)
            loss = torch.nn.CrossEntropyLoss()(output, target)

        # For logging accuracy with mixup/cutmix, convert soft targets back to hard labels.
        hard_target = target.argmax(dim=1) if target.ndim == 2 else target
        prec1 = accuracy(output.data, hard_target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    string = ('Test: [{0}][{1}/{2}]\t'
              'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
              'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
              'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\n'.format(
        epoch, (i + 1), train_batches_num, batch_time=batch_time,
        loss=losses, top1=top1))
    logging.info(string)

    val_acc.append(top1.ave)
    return top1.ave


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', save_path='./'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_path, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.ave = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.ave = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate.

    - If --cos_lr: linear warmup for args.warmup_epochs, then cosine decay to args.min_lr.
    - Else: step decay using training_configurations[...]['changing_lr'].
    """
    base_lr = training_configurations[args.model]['initial_learning_rate']
    total_epochs = training_configurations[args.model]['epochs']

    if not args.cos_lr:
        if epoch in training_configurations[args.model]['changing_lr']:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= training_configurations[args.model]['lr_decay_rate']
        print('lr:')
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        return

    # Cosine + warmup (DeiT-style)
    warmup = max(0, int(args.warmup_epochs))
    if warmup > 0 and epoch <= warmup:
        lr = base_lr * float(epoch) / float(warmup)
    else:
        # cosine from base_lr -> min_lr over the remaining epochs
        e = epoch - warmup
        T = max(1, total_epochs - warmup)
        lr = args.min_lr + 0.5 * (base_lr - args.min_lr) * (1.0 + math.cos(math.pi * e / T))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    print('lr:')
    print(lr)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def set_random_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        if args.device == 'cuda':
            torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    main()

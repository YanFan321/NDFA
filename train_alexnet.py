import argparse
import copy
import csv
import logging
import os
import shutil
import time
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
from torchvision.datasets import ImageFolder
from networks.alexnet import AlexNet
from utils.others import *
import json
import random
import torch.nn as nn
import torch.autograd.forward_ad as fwAD


parser = argparse.ArgumentParser(description='Implementation of AugLocal in PyTorch')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset: [cifar10|cifar100]')
parser.add_argument('--model', default='alexnet', type=str)
parser.add_argument('--droprate', default=0.0, type=float, help='dropout probability (default: 0.0)')
parser.add_argument('--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='', type=str, help='name of experiment')
parser.add_argument('--print-freq', '-p', default=10, type=int, help='print frequency (default: 10)')
parser.add_argument('--cos_lr', dest='cos_lr', action='store_true', default=True,
                    help='whether to use cosine learning rate')
parser.add_argument('--epochs', default=100, type=int, help='number of training epochs')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--hidden_dim', default=128, type=int)

parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--wd', default=5e-5, type=float, help='weight decay')
parser.add_argument('--save_path', default='', type=str, help='the directory used to save the trained models')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--train_mode', default='NDFA', type=str, help='the train_mode')
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


training_configurations = {
    'alexnet': {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'initial_learning_rate': args.lr,
        'changing_lr': [80, 120],
        'lr_decay_rate': 0.1,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': args.wd,
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


    if args.dataset in ['cifar10']:
        class_num = 10
    elif args.dataset in ['cifar100']:
        class_num = 100

    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    kwargs_dataset_train = {'train': True}
    kwargs_dataset_test = {'train': False}

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(pad_image),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    image_size = 32
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    kwargs = {'num_workers': 6, 'pin_memory': True}
    if args.dataset in ['imagenet']:
        train_dir = os.path.join("../image_datasets/", args.dataset, "train")
        val_dir = os.path.join("../image_datasets/", args.dataset, "val")
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
        batch_size=args.batch_size, shuffle=True, drop_last=False, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False, **kwargs)

    with open(save_path + '/args.json', 'w') as fid:
        json.dump(args.__dict__, fid, indent=2)

    logging.info('args: ' + str(args))

    # create model
    if args.model == 'alexnet':
        model = AlexNet(
            train_mode=args.train_mode,
            dim=args.hidden_dim,
            class_num=class_num,
            )
    else:
        raise NotImplementedError

    model.eval()

    cudnn.benchmark = True

    model = model.to(device)
    model = torch.nn.DataParallel(model)


    validate(val_loader, model, 0, None)
    logging.info(str(model))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        val_acc = checkpoint['val_acc']
        best_prec1 = checkpoint['best_acc']
    else:
        start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch + 1)

        # train for one epoch
        train_acc, loss_output  = train(
            train_loader,
            model,
            optimizer,
            epoch,
            None,
        )

        # evaluate on validation set
        prec1 = validate(val_loader, model, epoch, None)

        logging.info('train acc {:.4f} test acc {:.4f}'.format(train_acc, prec1))
        best_prec1 = max(prec1, best_prec1)
        logging.info(args.train_mode + args.dataset + ' Best accuracy: ' + str(best_prec1))
        if args.save:
            ckpt_file = os.path.join('./ckpt', args.model + args.train_mode + args.dataset + 'ckpt.pth.tar')
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': _checkpoint_state_dict(model),
                    'optimizer': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'best_acc': best_prec1,
                    'args': args.__dict__,
                },
                False,
                filename=ckpt_file,
                save_path=save_path,
            )
def train(train_loader, model, optimizer, epoch, writer):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    train_batches_num = len(train_loader)

    # switch to train mode
    model.train()
    net = model.module if isinstance(model, torch.nn.DataParallel) else model

    end = time.time()

    for i, (x, target) in enumerate(train_loader):
        if args.device == 'cuda':
            target = target.cuda(non_blocking=True)
            x = x.cuda(non_blocking=True)

        optimizer.zero_grad()

        outputs, loss = model(x, target)
        optimizer.step()

        prec1 = accuracy(outputs.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), target.size(0))
        top1.update(prec1.item(), target.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) == len(train_loader):
            string = ('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                      'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                      'Prec@1 {top1.value:.3f} ({top1.ave:.3f})'.format(
                epoch, i + 1, train_batches_num, batch_time=batch_time,
                loss=losses, top1=top1))
            if args.device == 'cuda':
                string += (
                    'mem={:.0f}MiB, max_mem={:.0f}MiB, max_mem={:.0f}MiB \n'.format(
                        torch.cuda.memory_allocated() / 1e6,
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

        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        with torch.no_grad():

            output, loss = model(input, target)

        prec1 = accuracy(output.data, target, topk=(1,))[0]
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
def _checkpoint_state_dict(model):
    state_dict = model.state_dict()
    if any(key.startswith('module.') for key in state_dict.keys()):
        return state_dict
    return {('module.' + key): value for key, value in state_dict.items()}

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
    """Sets the learning rate"""
    if not args.cos_lr:
        if epoch in training_configurations[args.model]['changing_lr']:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= training_configurations[args.model]['lr_decay_rate']
        print('lr:')
        for param_group in optimizer.param_groups:
            print(param_group['lr'])

    else:
        for param_group in optimizer.param_groups:
            if epoch <= -1:
                param_group['lr'] = 0.5 * args.lr \
                                    * (1 + math.cos(
                    math.pi * epoch / args.epochs)) * (epoch - 1) / 10 + 0.01 * (
                                            11 - epoch) / 10
            else:
                param_group['lr'] = 0.5 * args.lr \
                                    * (1 + math.cos(math.pi * epoch / args.epochs))
        print('lr:')
        for param_group in optimizer.param_groups:
            print(param_group['lr'])


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

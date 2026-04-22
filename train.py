import argparse
import csv
import copy
import logging
import math
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
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.autograd.forward_ad as fwAD
from torch import optim
from torchvision.datasets import ImageFolder

from networks.resnet32_imagenet import resnet32_imagenet
from utils.others import *
import json
import random
import networks.resnet
from networks.conv import ConvNet2FC2
from networks.alexnet import AlexNet
parser = argparse.ArgumentParser(description='Implementation of AugLocal in PyTorch')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset: [cifar10|cifar100|svhn]')
parser.add_argument('--model', default='resnet', type=str, help='model: [c2f2|resnet|alexnet]')
parser.add_argument('--layers', default=32, type=int, help='total number of layers (have to be explicitly given!)')
parser.add_argument('--droprate', default=0.0, type=float, help='dropout probability (default: 0.0)')
parser.add_argument('--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='', type=str, help='name of experiment')
parser.add_argument('--cos_lr', dest='cos_lr', action='store_true', default=False,
                    help='whether to use cosine learning rate')
parser.add_argument('--local_module_num', default=16, type=int,
                    help='number of local modules (1 refers to end-to-end training)')
parser.add_argument('--dfa_num', default=16, type=int)
parser.add_argument('--hidden_dim', default=128, type=int, help='number of hidden features in auxiliary classifier')
parser.add_argument('--epochs', default=200, type=int, help='number of training epochs')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--wd', default=5e-5, type=float, help='weight decay')
parser.add_argument('--save_path', default='', type=str, help='the directory used to save the trained models')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--train_mode', default='NDFA', type=str, help='the train_mode')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--device', default=str(device), type=str, help='the device')

args = parser.parse_args()
save_path = None


# Configurations adopted for training deep networks.
training_configurations = {
    'resnet': {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'initial_learning_rate': args.lr,
        'changing_lr': [80, 120],
        'lr_decay_rate': 0.1,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': args.wd,
    },
    'c2f2': {
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
    global save_path
    global device

    if args.save_path == '':
        local_save_path = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        local_save_path = local_save_path + args.name + '_' + str(args.seed)
    else:
        local_save_path = args.save_path
    save_path = local_save_path

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    setup_logging(os.path.join(save_path, 'log.txt'))
    logging.info('saving to:' + str(save_path))

    set_random_seed(seed=args.seed)
    global best_prec1
    best_prec1 = 0
    global val_acc
    val_acc = []

    if args.dataset == 'imagenet':
        class_num = 1000
    elif args.dataset in ['cifar10', 'svhn']:
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
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    elif 'cifar' in args.dataset:
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

    elif 'svhn' in args.dataset:
        transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=2),
             transforms.ToTensor(),
             normalize])
        image_size = 32
    else:
        raise NotImplementedError

    transform_test = transforms.Compose([
        transforms.Resize(256) if (args.dataset in ['imagenet', 'imagenette']) else transforms.Lambda(lambda x: x),
        transforms.CenterCrop(image_size) if (args.dataset in ['imagenet', 'imagenette']) else transforms.Lambda(
            lambda x: x),
        transforms.ToTensor(),
        normalize
    ])

    kwargs = {'num_workers': 6, 'pin_memory': True}
    if kwargs['num_workers'] > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = 4
    if args.dataset in ['imagenet', 'imagenette']:
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
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        **kwargs)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **kwargs)

    with open(save_path + '/args.json', 'w') as fid:
        json.dump(args.__dict__, fid, indent=2)

    logging.info('args: ' + str(args))

    # create model
    if args.model == 'resnet':
        if args.dataset == 'imagenet':
            model = resnet32_imagenet(
                train_mode=args.train_mode,
                hidden_dim=args.hidden_dim,
                class_num=class_num,
                dropout_rate=args.droprate,
            )
        else:
            model = eval('networks.resnet.resnet' + str(args.layers)) \
                (local_module_num=args.local_module_num,
                 class_num=class_num,
                 dfa_num=args.dfa_num,
                 wide_list=(16, 16, 32, 64),
                 dropout_rate=args.droprate,
                 hidden_dim=args.hidden_dim,
                 train_mode=args.train_mode,
                 )
    elif args.model == 'c2f2':
        model = ConvNet2FC2(
            train_mode=args.train_mode,
            dim=args.hidden_dim,
            class_num=class_num,
            )
    else:
        raise NotImplementedError
    model.eval()

    logging.info(str(model))

    cudnn.benchmark = True

    model = model.to(device)
    model = torch.nn.DataParallel(model)

    validate(val_loader, model, 0, None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    start_epoch = 0

    for epoch in range(start_epoch, training_configurations[args.model]['epochs']):
        adjust_learning_rate(optimizer, epoch + 1)

        # train for one epoch
        train_acc, loss_output = train(
            train_loader,
            model,
            optimizer,
            epoch,
            None,
        )

        # evaluate on validation set
        prec1 = validate(val_loader, model, epoch, None)
        logging.info('train acc {:.4f} test acc {:.4f}'.format(train_acc, prec1))
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        logging.info(args.train_mode +' Best accuracy: ' + str(best_prec1))


def train(train_loader, model, optimizer, epoch, writer):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    train_batches_num = len(train_loader)

    # switch to train mode
    model.train()


    end = time.time()
    for i, (x, target) in enumerate(train_loader):

        target = target.cuda(non_blocking=True)
        x = x.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        output, loss = model(x, target)
        optimizer.step()
        prec1 = accuracy(output.data, target, topk=(1,))[0]
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


        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        with torch.no_grad():
            output, loss = model(input,target)

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
                param_group['lr'] = 0.5 * training_configurations[args.model]['initial_learning_rate'] \
                                    * (1 + math.cos(
                    math.pi * epoch / training_configurations[args.model]['epochs'])) * (epoch - 1) / 10 + 0.01 * (
                                            11 - epoch) / 10
            else:
                param_group['lr'] = 0.5 * training_configurations[args.model]['initial_learning_rate'] \
                                    * (1 + math.cos(math.pi * epoch / training_configurations[args.model]['epochs']))
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
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    main()

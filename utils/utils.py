import torch
import math
import numpy as np
import torch.optim as optim
from torchvision import transforms
from models.ResNet import *
from torch.optim.lr_scheduler import _LRScheduler

cifar10_mean = (0.4914, 0.4823, 0.4466)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)

def get_architecture(args):
    if args.arch in ['MobileNet']:
        net = globals()[args.arch](args).to(args.device)
    elif args.arch in ['ResNet18','ResNet34','ResNet50','ResNet101']:
        net = globals()[args.arch](args).to(args.device)
    elif args.arch in ['WideResNet28_2','WideResNet28_10','WideResNet40_2','WideResNet40_4']:
        net = globals()[args.arch](args).to(args.device)
    elif args.arch in ['DenseNet']:
        net = globals()[args.arch](args).to(args.device)
    elif args.arch in ['EfficientNet']:
        pass
    return net

def get_optim_scheduler(args,net,epoch_per_step=None):
    if epoch_per_step==None:
        epoch_per_step=args.epoch


    if args.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    if args.optimizer == 'Nesterov':
        optimizer = optim.SGD(net.parameters(), lr = args.lr, momentum=0.9, nesterov= True, weight_decay=args.wd)
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr = args.lr)
    if args.optimizer == 'AdamW':
        optimizer = optim.AdamW(net.parameters(), lr = args.lr)

    if args.scheduler == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [int(epoch_per_step*0.5),int(epoch_per_step*0.75)],gamma=0.2)
    elif args.scheduler == 'CosineAnnealing':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epoch_per_step)
    elif args.scheduler == 'CosineWarmup':
        torch_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epoch_per_step)
        scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps = epoch_per_step, cycle_mult=1.0, max_lr = args.lr, min_lr = args.lr/100, warmup_steps = 10, gamma = 1.0)
    return optimizer, scheduler

def get_transform(args,mode):
    if args.in_dataset == 'cifar10':
        normalize = transforms.Normalize(mean = cifar10_mean, std = cifar10_std)
    elif args.in_dataset == 'cifar100':
        normalize = transforms.Normalize(mean = cifar100_mean, std = cifar100_std)
    elif args.in_dataset == 'svhn':
        normalize = transforms.Normalize(mean = cifar10_mean, std = cifar10_std)
        
    if mode == 'projection':
        train_TF = transforms.Compose([
            transforms.RandomResizedCrop(size = 32, scale = (0.2,1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4,0.4,0.4,0.1)
                ],p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
            ])
        test_TF = transforms.Compose([
        transforms.ToTensor(),
        normalize,
        ])
        return TwoCropTransform(train_TF), test_TF
    elif mode == 'linear':
        train_TF = transforms.Compose([
            transforms.RandomResizedCrop(size = 32, scale = (0.2,1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4,0.4,0.4,0.1)
                ],p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
            ])
        test_TF = transforms.Compose([
        transforms.ToTensor(),
        normalize,
        ])
        return train_TF, test_TF

    elif mode == 'eval':
        train_TF = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding = int(32*0.125)),
            transforms.ToTensor(),
            normalize,
        ])
        test_TF = transforms.Compose([
        transforms.ToTensor(),
        normalize,
        ])
        return train_TF, test_TF
    
class TwoCropTransform:
    "Create 2-way augmented images"
    def __init__(self,transform):
        self.transform = transform

    def __call__(self,x):
        return [self.transform(x),self.transform(x)]


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def count_layer(net):
    number_of_layer=0
    for child in net.children():
        if type(child)==torch.nn.modules.container.Sequential:
            number_of_layer+=1
    return number_of_layer

def freeze_model(net):
    for param in net.parameters():
        param.requires_grad = False

def enable_layer(net,steps):
    '''
    regards 'Sequential' layer as main points of CNN
    this method enables [steps]'th Sequential layer trainable
    '''
    layer_count=0
    for child in net.children():
        for param in child.parameters():
                param.requires_grad=True
        if type(child)==torch.nn.modules.container.Sequential:
            layer_count+=1
        if layer_count==steps:
            return 1
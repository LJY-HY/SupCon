import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
from tqdm import tqdm
import argparse
from utils.arguments import get_arguments
from utils.utils import *
from utils.loss import *
from dataset.cifar import *

def main():
    # Argument parsing
    args = argparse.ArgumentParser()
    args = get_arguments()
    args.device = torch.device('cuda',args.gpu_id)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    # dataset/transform setting
    if args.in_dataset in ['cifar10']:
        args.num_classes=10
    elif args.in_dataset in ['cifar100']:
        args.num_classes=100

    # Get Dataloader
    train_proj_dataloader, test_dataloader = globals()[args.in_dataset](args,mode = 'projection')
    train_linear_dataloader, test_dataloader = globals()[args.in_dataset](args, mode = 'linear')

    # Get architecture
    net = get_architecture(args)

    # Get optimizer, scheduler
    optimizer_proj, scheduler_proj = get_optim_scheduler(args,net)
    
    # Define Loss
    XentLoss = nn.CrossEntropyLoss()
    ContrastiveLoss = SupConLoss(temperature=0.1)

    # Projection Learning
    path = './checkpoint/'+args.in_dataset+'/'+args.arch+'_proj_'+str(args.wd)+'_trial_'+args.trial
    best_loss = 100.
    for epoch in range(args.epoch):
        projection_train(args, net, train_proj_dataloader, optimizer_proj, scheduler_proj, ContrastiveLoss, epoch)
        loss = test(args, net, test_dataloader, optimizer_proj, scheduler_proj, ContrastiveLoss, epoch,'SupCon')
        scheduler_proj.step()
        if best_loss>loss:
            best_loss = loss
            if not os.path.isdir('checkpoint/'+args.in_dataset):
                os.makedirs('checkpoint/'+args.in_dataset)
            torch.save(net.state_dict(), path)

    # Classifier Learning
    checkpoint = torch.load('./checkpoint/'+args.in_dataset+'/'+args.arch+'_proj_'+str(args.wd)+'_trial_'+args.trial)
    net.load_state_dict(checkpoint)
    net.train()

    # Freeze Network except linear layer for classification
    for para in net.parameters():
        para.requires_grad = False
    net.linear.weight.requires_grad = True
    net.linear.bias.requires_grad = True
    
    path = './checkpoint/'+args.in_dataset+'/'+args.arch+'_linear_'+str(args.wd)+'_trial_'+args.trial

    args.lr = 0.1

    # args.scheduler = 'MultiStepLR'
    args.wd = 1e-4
    args.epoch = 50
    optimizer_linear, scheduler_linear = get_optim_scheduler(args,net)

    best_acc = 0.
    for epoch in range(args.epoch):
        linear_train(args, net, train_linear_dataloader, optimizer_linear, scheduler_linear, XentLoss, epoch)
        acc = test(args, net, test_dataloader, optimizer_linear, scheduler_linear, XentLoss, epoch)
        scheduler_linear.step()
        if best_acc<acc:
            best_acc = acc
            if not os.path.isdir('checkpoint/'+args.in_dataset):
                os.makedirs('checkpoint/'+args.in_dataset)
            torch.save(net.state_dict(), path)

    print('Training End')
    acc = test(args, net, test_dataloader, optimizer_linear, scheduler_linear, XentLoss, 1)



def projection_train(args, net, train_proj_dataloader, optimizer, scheduler, SupConLoss, epoch):
    net.train()
    train_loss = 0
    p_bar = tqdm(range(train_proj_dataloader.__len__()))
    loss_average = 0
    for batch_idx, (images, labels) in enumerate(train_proj_dataloader):
        images = torch.cat([images[0],images[1]],dim=0)
        images, labels = images.to(args.device), labels.to(args.device)     
        optimizer.zero_grad()
        outputs = net(images, mode = 'SupCon')
        loss = SupConLoss(outputs,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        p_bar.set_description("Train Epoch: {epoch}/{epochs:2}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. loss: {loss:.4f}.".format(
                    epoch=epoch + 1,
                    epochs=args.epoch,
                    batch=batch_idx + 1,
                    iter=train_proj_dataloader.__len__(),
                    lr = scheduler.optimizer.param_groups[0]['lr'],
                    loss = train_loss/(batch_idx+1))
                    )
        p_bar.update()
    p_bar.close()
    return train_loss/train_proj_dataloader.__len__()        # average train_loss

def linear_train(args, net, train_linear_dataloader, optimizer, scheduler, XentLoss, epoch):
    net.train()
    train_loss = 0
    p_bar = tqdm(range(train_linear_dataloader.__len__()))
    loss_average = 0
    for batch_idx, (images, labels) in enumerate(train_linear_dataloader):
        images, labels = images.to(args.device), labels.to(args.device)     
        optimizer.zero_grad()
        outputs = net(images)
        loss = XentLoss(outputs,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        p_bar.set_description("Train Epoch: {epoch}/{epochs:2}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. loss: {loss:.4f}.".format(
                    epoch=epoch + 1,
                    epochs=args.epoch,
                    batch=batch_idx + 1,
                    iter=train_linear_dataloader.__len__(),
                    lr = [group['lr'] for group in scheduler.optimizer.param_groups],
                    loss = train_loss/(batch_idx+1))
                    )
        p_bar.update()
    p_bar.close()
    return train_loss/train_linear_dataloader.__len__()        # average train_loss

def test(args, net, test_dataloader, optimizer, scheduler, Loss, epoch, mode = 'Xent'):
    net.eval()
    test_loss = 0
    acc = 0
    p_bar = tqdm(range(test_dataloader.__len__()))
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_dataloader):
            images, labels = images.to(args.device), labels.to(args.device)
            outputs = net(images,mode)
            loss = Loss(outputs, labels)
            test_loss += loss.item()
            p_bar.set_description("Test Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Loss: {loss:.4f}.".format(
                    epoch=1,
                    epochs=1,
                    batch=batch_idx + 1,
                    iter=test_dataloader.__len__(),
                    lr = scheduler.optimizer.param_groups[0]['lr'],
                    loss=test_loss/(batch_idx+1)))
            p_bar.update()
            acc+=sum(outputs.argmax(dim=1)==labels)
    p_bar.close()
    acc = acc/test_dataloader.dataset.__len__()
    if mode == 'Xent':
        print('Accuracy :'+ '%0.4f'%acc )
        return acc
    elif mode == 'SupCon':
        print('Loss : '+'%0.4f'%(test_loss*256/test_dataloader.dataset.__len__()))
        return test_loss*256/test_dataloader.dataset.__len__()

if __name__ == '__main__':
    main()
import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description = 'Training Arguments')
    parser.add_argument('--gpu_id', default='0', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--in_dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100', 'SVHN', 'imagenet'], help = 'dataset choice')
    parser.add_argument('--arch', default = 'ResNet18', type=str, choices = ['MobileNet','DenseNet','ResNet18','ResNet34','ResNet50','ResNet101','WideResNet28_2','WideResNet28_10','WideResNet40_2','WideResNet40_4','EfficientNet'])
    parser.add_argument('--optimizer', default = 'Nesterov', type=str, choices = ['SGD','Nesterov','Adam','AdamW'])
    parser.add_argument('--lr', default = 0.1, type=float)
    parser.add_argument('--epoch', default=300, type=int, help='number of total epochs')
    parser.add_argument('--batch_size', default=256, type=int, choices=[64,128,256,512])
    parser.add_argument('--scheduler', default='CosineWarmup', type=str, choices=['MultiStepLR','CosineAnnealing','CosineWarmup'])
    parser.add_argument('--warmup_duration', default=10, type=int, help='length of warming')
    parser.add_argument('--wd', '--weight_decay','--wdecay', default=5e-4, type=float, choices=[5e-4,1e-2,1e-3,1e-4,1e-5,1e-6])
    parser.add_argument('--trial', default = '0', type=str)
    args = parser.parse_args()
    return args
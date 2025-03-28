import os
import torch
from torchvision import datasets, transforms
def create_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 设置数据集下载的目录
data_path = '../../Dataset'  # 替换为您的数据存储路径
create_if_not_exist(data_path)

transform = transforms.Compose([transforms.ToTensor()])


# FashionMnist
train_dataset = datasets.FashionMNIST(root=data_path, train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root=data_path, train=False, download=True, transform=transform)


# MNIST
train_dataset = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)


# USPS
usps_path = os.path.join(data_path,"usps")
create_if_not_exist(usps_path)
train_dataset = datasets.USPS(root=usps_path, train=True, download=True, transform=transform)
test_dataset = datasets.USPS(root=usps_path, train=False, download=True, transform=transform)
#
## USPS 偶尔抽风，不知道为啥下载不了，但是直接用浏览器是可以下载的，手动吧~
# "train": [
#     "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2",
#     "usps.bz2",
#     "ec16c51db3855ca6c91edd34d0e9b197",
# ],
# "test": [
#     "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2",
#     "usps.t.bz2",
#     "8ea070ee2aca1ac39742fdd1ef5ed118",
# ],

# SVHN
svhn_path = os.path.join(data_path,"svhn")
create_if_not_exist(svhn_path)
train_dataset = datasets.SVHN(root=svhn_path, split="train", download=True, transform=transform)
test_dataset = datasets.SVHN(root=svhn_path, split="test", download=True, transform=transform)

# CIFAR10
cifar10_train_dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
cifar10_test_dataset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)


# CIFAR100
cifar100_train_dataset = datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform)
cifar100_test_dataset = datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform)
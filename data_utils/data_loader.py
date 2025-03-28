import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, SVHN, USPS
from PIL import Image

class MyDigits(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, data_name=None):
        self.data_name = data_name
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.dataset = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        if self.data_name == 'mnist':
            dataobj = MNIST(self.root + self.data_name, self.train, self.transform, self.target_transform, self.download)
        elif self.data_name == 'usps':
            dataobj = USPS(self.root + self.data_name, self.train, self.transform, self.target_transform, self.download)
        elif self.data_name == 'svhn':
            if self.train:
                dataobj = SVHN(self.root + self.data_name, 'train', self.transform, self.target_transform, self.download)
            else:
                dataobj = SVHN(self.root + self.data_name, 'test', self.transform, self.target_transform, self.download)
        return dataobj

    def __getitem__(self, index):
        img, target = self.dataset[index]
        img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
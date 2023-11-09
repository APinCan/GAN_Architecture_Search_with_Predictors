# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0
import os
from PIL import Image
import numpy as np
import pickle
import os.path

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class ImageNet32Dataset(torch.utils.data.Dataset):
    def __init__(self, root, train, transform):
        self.train_data_batch_files = ['train_data_batch_1', 'train_data_batch_2', 'train_data_batch_3', 'train_data_batch_4',
            'train_data_batch_5', 'train_data_batch_6', 'train_data_batch_7', 'train_data_batch_8', 'train_data_batch_9', 'train_data_batch_10'
        ]
        self.root = root
        self.transform = transform
        self.val_data = ['val_data']
        self.base_folder = 'SmallImageNet_32x32'

        self.train = train
        if self.train:
            data_list = self.train_data_batch_files
        else:
            data_list = self.val_data

        self.data = []
        self.targets = []

        for file_name in data_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f)
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


class ImageDataset(object):
    def __init__(self, args, cur_img_size=None):
        img_size = cur_img_size if cur_img_size else args.img_size
        if args.dataset.lower() == 'cifar10':
            Dt = datasets.CIFAR10
            transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            args.n_classes = 10
        elif args.dataset.lower() == 'stl10':
            Dt = datasets.STL10
            transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        elif args.dataset.lower() == 'cifar100':
            Dt = datasets.CIFAR100
            transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        elif args.dataset.lower() == 'imagenet32':
            transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            raise NotImplementedError('Unknown dataset: {}'.format(args.dataset))

        if args.dataset.lower() == 'stl10':
            self.train = torch.utils.data.DataLoader(
                Dt(root=args.data_path, split='train+unlabeled', transform=transform, download=True),
                batch_size=args.dis_batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True)

            self.valid = torch.utils.data.DataLoader(
                Dt(root=args.data_path, split='test', transform=transform),
                batch_size=args.dis_batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True)

            self.test = self.valid
        elif args.dataset.lower()=='imagenet32':
            self.train = torch.utils.data.DataLoader(
                ImageNet32Dataset(root=args.data_path, train=True, transform=transform),
                batch_size=args.dis_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
            self.train = torch.utils.data.DataLoader(
                ImageNet32Dataset(root=args.data_path, train=False, transform=transform),
                batch_size=args.dis_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        else:
            self.train = torch.utils.data.DataLoader(
                Dt(root=args.data_path, train=True, transform=transform, download=True),
                batch_size=args.dis_batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True)

            self.valid = torch.utils.data.DataLoader(
                Dt(root=args.data_path, train=False, transform=transform),
                batch_size=args.dis_batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True)

            self.test = self.valid


class JustGenerated(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image
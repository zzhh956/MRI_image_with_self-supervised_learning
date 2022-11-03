import os
import cv2
import tqdm
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms.functional import rotate

class ImageLoader(Dataset):
    def __init__(self, mode = 'train'):
        super().__init__()
        self.mode = mode
        self.train_path = Path('../unlabeled/')
        self.test_path = Path('../test/')

        if self.mode == 'train':
            self.dir = self.train_path
        else:
            self.dir = self.test_path
        self.img_path_list = sorted(self.dir.rglob("*.jpg"))
        
        self.size = 96

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        path = str(self.img_path_list[index])

        # width x height target sizes: [96, 96, 3]
        img = cv2.imread(path)

        # change color sequence
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = img / 255.0
        # frame = np.mean(frame, axis = 2)

        img = self.bytetotensor(img)
            
        sample = {}

        # image tensor -> batch_size x color(rgb) x width x height 
        # sample['image'] = torch.permute(img, (0, 1, 2))
        sample['image'] = img
        # print(sample['image'].shape)

        if self.mode == 'test':
            # test label
            dirname = os.path.dirname(self.img_path_list[index])
            path = os.path.split(dirname)[1]
            sample['label'] = torch.as_tensor(int(path))

        return sample

    def bytetotensor(self, x):
        transform = T.Compose([
            T.ToTensor(), # range [0, 255] -> [0.0,1.0]
            ]
        )

        return transform(x)

    def Augment(self, x):
        s = 1
        color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)

        # 10% of the image
        blur = T.GaussianBlur((3, 3), (0.1, 2.0))

        train_transform = torch.nn.Sequential(
            T.RandomResizedCrop(size = self.size),
            T.RandomHorizontalFlip(p = 0.5),  # with 0.5 probability
            T.RandomApply([color_jitter], p = 0.8),
            T.RandomApply([blur], p = 0.5),
            T.RandomGrayscale(p = 0.2),
        )

        return train_transform(x), train_transform(x)
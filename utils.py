import os
import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets


def edge_promoting(src_path, save_path):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    img_loader = torch.utils.data.DataLoader(
        src_path, batch_size=1, shuffle=True, num_workers=0)
    for i, data in enumerate(img_loader):



        cv2.imwrite(os.path.join(save_path, str(i) + '.png'), data)
        
        
# class MyDataset(torch.utils.data.Dataset):
#     def __init__(self, path, transform):
#         super(MyDataset, self).__init__():
        
#         self.path = path
#         self.path_list = glob.glob(os.path.join(self.path, '*'))
#         # get transform
#         self.transform = transform
        
def data_load(path, subfolder, transform, batch_size, shuffle=False, drop_last=True):
    dset = datasets.ImageFolder(path, transform)
    ind = dset.class_to_idx[subfolder]

    n = 0
    for i in range(dset.__len__()):
        if ind != dset.imgs[n][1]:
            del dset.imgs[n]
            n -= 1

        n += 1

    return torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        
        
        
        
        
        
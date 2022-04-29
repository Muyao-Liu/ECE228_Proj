import os
import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms




def edge_promoting(src_path, save_path):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    img_loader = torch.utils.data.DataLoader(
        src_path, transform, batch_size=1, shuffle=True, num_workers=0)
    for i, data in enumerate(img_loader):



        cv2.imwrite(os.path.join(save_path, str(i) + '.png'), data)
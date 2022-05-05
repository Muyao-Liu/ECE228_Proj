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
       
    #new added
    file_list=glob.glob(os.path.join(src_path,'*'))
    
    kernel_size=5
    kernel=np.ones((kernel_size,kernel_size),np.uint8)
    gauss=cv2.getGaussianKernel(kernel_size,0)
    gauss=gauss*gauss.transpose(1,0)


    '''transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    img_loader = torch.utils.data.DataLoader(
        src_path, batch_size=1, shuffle=True, num_workers=0)
    for i, data in enumerate(img_loader):



        cv2.imwrite(os.path.join(save_path, str(i) + '.png'), data)'''
    
    n=1
    #img_loader = torch.utils.data.DataLoader(src_path, batch_size=1, shuffle=True, num_workers=0)
    
    for f in tqdm(file_list):
        #read origin image
        #rgb_img=cv2.imread(os.path.join(src_path,str(i)+'_A.png'))
        #rgb_img=cv2.imread(os.path.join(src_path,f))
        rgb_img=cv2.imread(f)
        gray_img=cv2.imread(f,0)
        #gray_img=cv2.imread(os.path.join(src_path,f),0)
        
        try:
            #resize and padding
            rgb_img=cv2.resize(rgb_img,(256,256))
            pad_img=np.pad(rgb_img,((2,2),(2,2),(0,0)),mode='reflect')
            gray_img=cv2.resize(gray_img,(256,256))
            #process edge
            edges=cv2.Canny(gray_img,100,200)
            dilation=cv2.dilate(edges,kernel)
            
            gauss_img=np.copy(rgb_img)
            idx=np.where(dilation!=0)
            for i in range(np.sum(dilation!=0)):
                #process the third channel only
                gauss_img[idx[0][i],idx[1][i],0]=np.sum(np.multiply(pad_img[idx[0][i]:idx[0][i]+kernel_size,idx[1][i]:idx[1][i]+kernel_size,0],gauss))
                gauss_img[idx[0][i],idx[1][i],1]=np.sum(np.multiply(pad_img[idx[0][i]:idx[0][i]+kernel_size,idx[1][i]:idx[1][i]+kernel_size,1],gauss))
                gauss_img[idx[0][i],idx[1][i],2]=np.sum(np.multiply(pad_img[idx[0][i]:idx[0][i]+kernel_size,idx[1][i]:idx[1][i]+kernel_size,2],gauss))
                
                res=np.concatenate((rgb_img,gauss_img),1)
                cv2.imwrite(os.path.join(save_path,str(n)+'.png'),res)
                n=n+1
        
        except Exception as e:
            print(str(e))
        
        
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
        
        
        
        
        
        

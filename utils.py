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
        
    index=1 
    
    #get files from source path
    get_file=glob.glob(os.path.join(src_path,'*'))
    
    #initiate Gaussian kernel
    gauss_kernel=np.ones((3,3),np.uint8)
    gauss=cv2.getGaussianKernel(3,0)
    gauss=gauss*gauss.transpose(1,0)
    
    for file in tqdm(get_file):
        #read in the gray image and rgb image
        gray=cv2.imread(file,0)
        rgb=cv2.imread(file)
    
        #by pass the empty dimension
        try:
            #resize and padding
            uni_size=256
            pad_size=2
            empty_pad=0
            canny_thr1=100
            canny_thr2=200
            
            uni_gray=cv2.resize(gray,(uni_size,uni_size))
            uni_rgb=cv2.resize(rgb,(uni_size,uni_size))
            
            rgb_pad=np.pad(uni_rgb,((pad_size,pad_size),(pad_size,pad_size),(empty_pad,empty_pad)),mode='reflect')
            
            #detect edges with canny
            detected_edges=cv2.Canny(uni_gray,canny_thr1,canny_thr2)
            
            #dilate edges with Gaussian kernel
            dilation=cv2.dilate(detected_edges,gauss_kernel)
            res_img=np.copy(uni_rgb)
            
            #dilate pixel whose index is not equal to 0
            dilate_index=np.where(dilation!=0)
            index_len=len(dilate_index[0])
            
            
            for i in range(index_len):
                res_img[dilate_index[0][i],dilate_index[1][i],0]=np.sum(np.multiply(rgb_pad[dilate_index[0][i]:dilate_index[0][i]+3,dilate_index[1][i]:dilate_index[1][i]+3,0],gauss))
                res_img[dilate_index[0][i],dilate_index[1][i],1]=np.sum(np.multiply(rgb_pad[dilate_index[0][i]:dilate_index[0][i]+3,dilate_index[1][i]:dilate_index[1][i]+3,1],gauss))
                res_img[dilate_index[0][i],dilate_index[1][i],2]=np.sum(np.multiply(rgb_pad[dilate_index[0][i]:dilate_index[0][i]+3,dilate_index[1][i]:dilate_index[1][i]+3,2],gauss))
                
                #easy to compare result
                output=np.concatenate((uni_rgb,res_img),1)
                
                
            cv2.imwrite(os.path.join(save_path,str(index)+'.png'),output)
            #cv2.imwrite(save_path,output)
            
        except Exception as e:
            print(str(e))
        
        
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
        
        
        
        
        
        

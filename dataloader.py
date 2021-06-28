import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import os
import torch


class FloorPlanDataset(Dataset):
    def __init__(self,image_dir,gt_dir,transform=None):
        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return(len(self.images))

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir,self.images[index])
        
        
        gt_path = os.path.join(self.gt_dir,self.images[index])
        gt_path = gt_path.replace(".jpg",".png")

        image = Image.open(image_path).convert("RGB")
        image = image.resize((600,600),Image.ANTIALIAS)
        image = np.array(image,dtype=np.float32)
        
        gt = Image.open(gt_path).convert("L")
        gt = np.array(gt,dtype=np.float32)

        gt[np.all(gt == 0)] = 0 #black background
        gt[np.all(gt == 127)] = 1 #green windows
        gt[np.all(gt == 38)] = 2 #blue doors

        image = image.reshape([image.shape[-1],image.shape[0],image.shape[1]])
        #gt = gt.reshape([gt.shape[-1],gt.shape[0],gt.shape[1]])
        

        #plt.imsave(self.images[index],arr=gt/255)       
        if self.transform is not None:
            image = torch.tensor([image])
            gt = torch.tensor([gt])
        
        #print(np.shape(image),np.shape(gt))
        
        return image, gt
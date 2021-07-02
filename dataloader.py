import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch._C import dtype
from torch.utils.data import Dataset
import os
import torch


class FloorPlanDataset(Dataset):
    def __init__(self,image_dir,gt_dir,transform=False):
        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return(len(self.images))

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir,self.images[index])
        
        
        gt_path = os.path.join(self.gt_dir,self.images[index])
        gt_path = gt_path.replace(".jpg","_windows.png")

        image = Image.open(image_path).convert("RGB")
        image = image.resize((600,600),Image.ANTIALIAS)
        image = np.array(image,dtype=np.float32)
        
        gt = Image.open(gt_path).convert("L")
        gt = np.array(gt,dtype=np.long)

        #image = image/255.0
        gt = gt/255.0

        gt[np.all(gt == 0.0)] = 0 #black background
        gt[np.all(gt == 0.498)] = 1 #green windows
        gt[np.all(gt == 0.149)] = 2 #blue doors
        #gt[np.all(gt == 1.0)] = 1

        image = np.transpose(image, (2,0,1))
        #gt = gt.reshape([1,gt.shape[0],gt.shape[1]])

        #plt.imsave(self.images[index],arr=gt/255)       
        if self.transform is True:
            image = torch.tensor([image])
            gt = torch.tensor([gt])
        
        gt = gt.type(torch.LongTensor)
        #print(np.shape(image),np.shape(gt))
        
        return image, gt
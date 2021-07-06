import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch._C import dtype
from torch.utils.data import Dataset
import os
import torch

SEG_LABELS_LIST = [
    {"id":-1,"name":"void","rgb_values":[0,0,0]},
    {"id":0,"name":"wall","rgb_values":[1,1,1]}
]

def label_img_to_rgb(label_img):
    label_img =  np.squeeze(label_img)
    labels = np.unique(label_img)
    label_infos = [l for l in SEG_LABELS_LIST if l['id'] in labels]
    label_img_rgb = np.array([label_img,label_img,label_img]).transpose(1,2,0)
    for l in label_infos:
        mask = label_img == l['id']
        label_img_rgb[mask] = l['rgb_values']
    
    return label_img_rgb.astype(np.uint8)

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
        image = np.array(image,dtype=np.float32).transpose(2,0,1)
        
        gt = Image.open(gt_path).convert("L")
        gt = np.array(gt,dtype=np.float32).transpose(2,0,1)
        
        """
        target_labels = gt[...,0]
        for label in SEG_LABELS_LIST:
            mask = np.all(gt == label['rgb_values'],axis=2)
            #print(mask)
            target_labels[mask] = label['id']
        """
        gt[np.all(gt == (0.0))] = 0 #black background
        #gt[np.all(gt == 0.498)] = 1 #green windows
        #gt[np.all(gt == 0.149)] = 2 #blue doors
        gt[np.all(gt == (1.0))] = 1
        
        
        #plt.imsave(self.images[index],arr=gt/255)       
        if self.transform is True:
            image = torch.tensor([image])
            gt = torch.tensor([gt])

        #print(torch.min(target_labels),torch.max(target_labels))
        #gt = gt.type(torch.LongTensor)
        #print(np.shape(image),np.shape(gt))
        
        return image, gt, self.images[index]
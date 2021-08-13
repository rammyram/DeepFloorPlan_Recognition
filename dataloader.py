import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import os
import torch
import cv2

SEG_LABELS_LIST = [
    {"id":-1,"name":"void","rgb_values":[0,0,0]},
    {"id":0,"name":"wall","rgb_values":[255,0,0]},
    {"id":1,"name":"door","rgb_values":[0,255,0]},
    {"id":2,"name":"window","rgb_values":[0,0,255]}
]

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
        gt_path = gt_path.replace(".jpg","")

        floor_plan = plt.imread(image_path)
        floor_plan_resized = cv2.resize(floor_plan,(600,600))
        floor_plan_resized /= floor_plan_resized.max()
        floor_plan = torch.tensor([floor_plan_resized])
        

        gt = plt.imread(gt_path) 

        gt_labels = gt[...,0]
        for label in SEG_LABELS_LIST:
            mask = np.all(gt == label["rgb_values"],axis=2)
            gt_labels[mask] = label["id"]
        
        gt_labels = torch.from_numpy(gt_labels.copy())
 
        return floor_plan, gt
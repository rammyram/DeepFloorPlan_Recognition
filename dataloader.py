import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os
import torch

class FloorPlanDataset(Dataset):
    def __init__(self,image_dir,door_dir,window_dir,transform=None):
        self.image_dir = image_dir
        self.door_dir = door_dir
        self.window_dir = window_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return(len(self.images))

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir,self.images[index])
        door_path = os.path.join(self.door_dir,self.images[index].replace('.jpg','_doors.png'))
        window_path = os.path.join(self.window_dir,self.images[index].replace('.jpg','_windows.png'))

        image = Image.open(image_path).convert('L')
        image = image.resize((320,320),Image.ANTIALIAS)
        image = np.array(image,dtype=np.float32)
        
        doors = Image.open(door_path).convert('L')
        doors = doors.resize((320,320),Image.ANTIALIAS)
        doors = np.array(doors,dtype=np.float32)
        
        windows = Image.open(window_path).convert('L')
        windows = windows.resize((320,320),Image.ANTIALIAS)
        windows = np.array(windows,dtype=np.float32)
        

        #doors[doors == 255.0] = 1.0
        #windows[windows == 255.0] = 1.0
        
        if self.transform is not None:
            image = torch.tensor([image])
            doors = torch.tensor([doors])
            windows = torch.tensor([windows])
        
        return image, doors, windows
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

        image = np.array(Image.open(image_path).convert("RGB"))
        doors = np.array(Image.open(door_path).convert("L"),dtype=np.float32)
        windows = np.array(Image.open(window_path).convert("L"),dtype=np.float32)

        doors[doors == 255.0] = 1.0
        windows[windows == 255.0] = 1.0

        if self.transform is not None:
            image = torch.tensor([image])
            doors = torch.tensor([doors])
            windows = torch.tensor([windows])

        return image, doors, windows
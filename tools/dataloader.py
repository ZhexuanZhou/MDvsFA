import glob
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

class MyDataset(Dataset):
    def __init__(self, data_root, mask_root, transform=None):
        self.data_paths = sorted(glob.glob(data_root)) 
        self.mask_paths = sorted(glob.glob(mask_root))
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        mask = Image.open(self.mask_paths[index])
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        
        return img, mask

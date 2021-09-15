import os
import numpy as np
import torch
import cv2
from natsort import natsorted
from torch.utils.data import Dataset as BaseDataset

img_dims_width = 512
img_dims_height = 512

class Dataset(BaseDataset):
    
    CLASSES = ['non-crack', 'crack']
    def __init__(self, images_dir, masks_dir, classes=['crack'], transform=None):
        self.train_ids = natsorted(next(os.walk(images_dir))[2])
        self.mask_ids = natsorted(next(os.walk(masks_dir))[2])

        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.train_ids]
        self.masks_fps = [os.path.join(masks_dir, mask_id) for mask_id in self.mask_ids]
        
        self.class_values= [self.CLASSES.index(cls.lower()) for cls in classes]
        self.transform = transform

        self.img_rows = img_dims_width
        self.img_cols = img_dims_height
        self.img_width = img_dims_width
        self.img_height = img_dims_height
    
    def __len__(self):
        return len(self.train_ids)
    
    def __getitem__(self, i):
        
        image = cv2.imread(self.images_fps[i])
        mask = cv2.imread(self.masks_fps[i],0)
        
        image = cv2.resize(image, dsize=(self.img_width, self.img_height))
        mask = cv2.resize(mask, dsize=(self.img_width, self.img_height)) 
        if self.transform:
            img, mask = self.transform(image), self.transform(mask)
        masks = [(mask==v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        image = torch.from_numpy(image).permute(2,0,1).float()
        mask = torch.from_numpy(mask).permute(2,0,1).float()
        return image, mask
    


            

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from model import VGG
from data import  Dataset
import os
from torch.utils.data import DataLoader

epochs = 1

if __name__ == "__main__":
    
    use_cuda = torch.cuda.is_available()
    
    DATA_DIR = "/media/preethamam/Utilities-SSD/Xtreme_Programming/Z_Data/DLCrack/Liu+Xincong+DS3+CrackSegNet"
    images_dir = os.path.join(DATA_DIR , "TrainingCracks")
    masks_dir = os.path.join(DATA_DIR , "TrainingCracksGroundtruth")
    train_data = Dataset(images_dir=images_dir, masks_dir = masks_dir)
    training_dataloader = DataLoader(train_data, batch_size=8, shuffle = True)
    model = VGG()
    #for i, data in enumerate(training_dataloader,0):
        #inputs, labels = data
        #optimizer.zero_grad()
        #outputs = model(inputs)
        #loss = criterion(outputs, labels)
        #loss.backward()
        #optimizer.step()

        #running_loss = loss.item()
        #if i % 2000
    #print("Finished Training")

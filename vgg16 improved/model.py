import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
from torchsummary import summary

class VGG(nn.Module):
    def __init__(self, n_classes):
        super(VGG, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64,kerne_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding = 1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding = 1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding = 1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        
        self.conv3_1_1 = nn.Conv2d(256, 64, kernel_size=1, padding=1)
        self.conv3_2_1 = nn.Conv2d(256, 64, kernel_size=1, padding=1)
        self.conv3_3_1 = nn.Conv2d(256, 64, kernel_size=1, padding=1)
        self.conv3_1_3 = nn.Conv2d(64, 12, kernel_size = 1, padding = 1)
        self.conv3_1_4 = nn.Conv2d(12, 1, kernel_size = 1, padding=1)
        
        self.conv5_1_1 = nn.Conv2d(512, 64, kernel_size=1, padding=1)
        self.conv5_2_1 = nn.Conv2d(512, 64, kernel_size=1, padding=1)
        self.conv5_3_1 = nn.Conv2d(512, 64, kernel_size=1, padding=1)


        self.pool = nn.MaxPool2d(2, 2)
        self.fc6 = nn.Linear(7*7*512, 4096)
        self.fc7 = nn.Linead(4096, 4096)
        self.fc8 = nn.Linear(4096, 1000)

    def forward(self, x, training=True):
        x_1_1 = F.relu(self.conv1_1(x))
        x_1_2 = F.relu(self.conv1_2(x_1_1))
        x_1_pool = self.pool(x_1_2)
        
        x_2_1 = F.relu(self.conv2_1(x_1_pool))
        x_2_2 = F.relu(self.conv2_2(x_2_1))
        x_2_pool = self.pool(x_2_2)
        
        x_3_1 = F.relu(self.conv3_1(x_2_pool))
        x_3_2 = F.relu(self.conv3_2(x_3_1))
        x_3_3 = F.relu(self.conv3_3(x_3_2))
        x_3_pool = self.pool(x_3_3)

        x_4_1 = F.relu(self.conv4_1(x_3_pool))
        x_4_2 = F.relu(self.conv4_2(x_4_1))
        x_4_3 = F.relu(self.conv4_3(x_4_2))
        x_4_pool = self.pool(x_4_3)
        
        x_5_1 = F.relu(self.conv5_1(x_4_pool))
        x_5_2 = F.relu(self.conv5_2(x_5_1))
        x_5_3 = F.relu(self.conv5_3(x_5_2))
        x_5_pool = self.pool(x_5_3)
        
        x_3_1_1 = F.relu(self.conv3_1_1(x_3_1))
        x_3_2_1 = F.relu(self.conv3_2_1(x_3_2))
        x_3_3_1 = F.relu(self.conv3_3_1(x_3_3))

        x_3_1_2 = x_3_1_1 + x_3_2_1 + x_3_3_1
        
        x_5_1_1 = F.relu(self.conv5_1_1(x_5_1))
        x_5_2_1 = F.relu(self.conv5_2_1(x_5_2))
        x_5_3_1 = F.relu(self.conv5_3_1(x_5_3))
        

        x_3_1_2 = torch.sum(x_3_1_1, x_3_2, x_3_3)
        x_5_1_2 = torch.sum(x_5_1_1, x_2_1, x_3_1)

        x_3_1_3 = f.relu(self.conv3_1_3(x_1_2))
        
        #x = x.view(-1, 7 * 7 * 512)
        #x = F.relu(self.fc6(x))
        #x = F.dropout(x, 0.5, training=training)
        #x = F.relu(self.fc7(x))
        #x = F.dropout(x, 0.5, training=training)
        #x = self.fc8(x)
        return x_3_1_3

    def predict(self, x):
        x = F.softmax(self.forward(x, training=False))
    


# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 01:29:23 2020

@author: Kamaljeet
"""

import cv2
import numpy as np
import torch
from Model import Net
import torchvision.transforms as transforms


#=========Setting up==============
#=======Loading weight file to the model========
from Dataset import load_split_train_test
from Model import Net

classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','del','nothing','space']

model = Net(out_fea = len(classes))

PATH = 'weight/epoch_6loss_0.15457870066165924.pt'
model.load_state_dict(torch.load(PATH))
model = model.eval()

img = cv2.imread('P:/Hand-Symbol-Recognition/asl_alphabet_test/asl_alphabet_test/J_test.jpg')
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
img = trans(img)
img = img.unsqueeze(0)

with torch.no_grad():
    output = model(img)
_, predicted = torch.max(output, 1)

print(classes[predicted])
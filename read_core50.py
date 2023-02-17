import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import random
import torchvision
from torch.utils.data import Dataset 
# Define simple dataset class in pytorch
#'/home/francesco/data/core50_128x128/'

class Dataset(Dataset):
    def __init__(self, data_path, scenario_number, transform=None):
        self.data_path = data_path
        self.scenario_number = scenario_number
        self.transform = transform
        
        self.paths = glob.glob(self.data_path+'/'+f's{scenario_number}/*/*.png')
        self.labels = self.get_labels_from_path(self.paths)

        # Shuffle the lists in unison
        combined = list(zip(self.paths, self.labels))
        random.shuffle(combined)
        self.paths , self.labels = zip(*combined)

    def get_labels_from_path(self, paths):
        labels = []
        for path in paths:
            labels.append(int(path.split('/')[-2][1:]))
        return labels
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        x = cv2.imread(self.paths[index])
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        y = self.labels[index]
        if self.transform:
            x = self.transform(x)

        return x, y
    

if __name__ == '__main__':
    for i in range(1,5):
        dataset = Dataset(data_path='/home/francesco/data/core50_128x128/', scenario_number=i)
        print(dataset[0][0].shape)
        print(dataset[0][1])
        print(len(dataset))



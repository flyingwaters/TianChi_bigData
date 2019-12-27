# @2019
# @content
# @fenglongyu
from preprocessing import*
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class poetrySet(Dataset):

    def __init__(self, train_x, train_y):
        self.train = torch.from_numpy(train_x)
        self.label = torch.from_numpy(train_y)

    def __getitem__(self, item):
        self.train = self.train
        self.label = self.label
        return self.train[item], label[item]
        #

    def __len__(self):
        return len(self.train)

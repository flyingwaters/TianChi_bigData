# @ fenglongyu
# @ 2019
# @ content: for the digging_for_happiness digging
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from sklearn.decomposition import PCA

#  PCA+BP+regression 10folds
#  PCA+KNN~Lazy Algorithm
#

class Gru(nn.Module):
    ###
    def __init__(self, input_size=90):
        super(Gru, self).__init__()
        # 降维
        self.linear_1 = nn.Linear(input_size, 50)
        self.linear_2 = nn.Linear(50, 25)
        self.bn = nn.BatchNorm1d(25)
        self.linear_3 = nn.Linear(25, 1)
        #

    def forward(self, x):
        # x = torch.tensor()
        out = self.linear_1(x)
        out = F.relu(out)
        out = self.linear_2(out)
        out = F.relu(out)
        out = self.bn(out)
        out = self.linear_3(out)
        x = F.sigmoid(out)
        x = x*5
        return x







import numpy as np
from torch.utils.data import Dataset
class CustomDataset(Dataset):
    def __init__(self, X , y):
       self.X = X
       self.y = y
    def __len__(self):
        y = self.y
        return y.size(0)
    def __getitem__(self, idx):                 
        return self.X[idx],self.y[idx]

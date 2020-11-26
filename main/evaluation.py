import numpy as np
import umap
import torchvision
import torch
import seaborn as sns
import os
import matplotlib.pyplot as plt
import sklearn

from dataloading import load_data, MiraBest_full

# define paths for saving
FILE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_PATH = os.path.join(FILE_PATH, 'files', 'eval')
DATA_PATH = os.path.join(FILE_PATH, 'data')
CHECKPOINT_PATH = os.path.join(FILE_PATH, 'files', 'checkpoints')

cuda = torch.device('cuda')

def dset_array():
    ## load and normalise data ## 
    transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    train_data = MiraBest_full(DATA_PATH, train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=len(train_data), shuffle=True)
    X, y = next(iter(train_loader))
    X, y = X.numpy(), y.numpy()
    return X, y

def class_idx(y):
    fri_idx = np.where(y < 4.5)
    frii_idx = np.where((y > 4.5) & (y < 7.5))
    hybrid_idx = np.where(y>7.5)
    return fri_idx, frii_idx, hybrid_idx

def E_map():
    E = torch.load(CHECKPOINT_PATH + '/E_250.pt')

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

def inception_score(G, n_z, n_samples=1000, eps = 1E-10):
    # I = torch.load()
    Z = torch.randn(n_samples, n_z).cuda().view(n_samples, n_z, 1, 1)
    X = G(Z)
    # p_y|x = I(X)
    # p_y = torch.mean(p_y|

def umap_map():
    X, y = dset_array()
    X = np.reshape(X, (-1, 150**2))
    fri_idx = np.where(y < 4.5)
    frii_idx = np.where((y > 4.5) & (y < 7.5))
    hybrid_idx = np.where(y>7.5)

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(X)
    embedding = sklearn.preprocessing.StandardScaler().fit_transform(embedding)

    plt.scatter(embedding[fri_idx, 0], embedding[fri_idx, 1], c='red', label='fri')
    plt.scatter(embedding[frii_idx, 0], embedding[frii_idx, 1], c='blue', label='frii')
    plt.scatter(embedding[hybrid_idx, 0], embedding[hybrid_idx, 1], c='green', label='hybrid')
    plt.legend()
    plt.savefig(EVAL_PATH + '/umap.pdf')

def E_map():
    E = torch.load(CHECKPOINT_PATH + '/E_250.pt')

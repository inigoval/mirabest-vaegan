import numpy as np
import umap
import torchvision
import torch
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from dataloading import load_data, MiraBest_full

# define paths for saving
FILE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EVAL_PATH = os.path.join(FILE_PATH, 'files', 'eval')
DATA_PATH = os.path.join(FILE_PATH, 'data')
CHECKPOINT_PATH = os.path.join(FILE_PATH, 'files', 'checkpoints')
FIG_PATH = os.path.join(FILE_PATH, 'files', 'figs')

EMBEDDING_PATH = os.path.join(EVAL_PATH, 'embeddings')

cuda = torch.device('cuda')

def generate(G, n_z, n_samples=1000):
    Z = torch.randn(n_samples, n_z).cuda().view(n_samples, n_z, 1, 1)
    X = G(Z)
    return X

def inception_score(X, eps = 1E-10):
    I = torch.load(EVAL_PATH + '/I.pt')
    p_yx = I(X)
    p_y = torch.mean(p_yx, 0)
    KL = torch.mean(p_yx * (torch.log(p_yx + eps) - torch.log(p_y + eps)))
    return KL

def dset_array():
    ## load and normalise data ## 
    transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    train_data = MiraBest_full(DATA_PATH, train=True, transform=transform, download=True)
    test_data = MiraBest_full(DATA_PATH, train=False, transform=transform, download=True)
    all_data = torch.utils.data.ConcatDataset((train_data, test_data))
    train_loader = torch.utils.data.DataLoader(all_data, batch_size=len(train_data), shuffle=True)
    X, y = next(iter(train_loader))
    X, y = X.numpy(), y.numpy()
    return X, y

def E_map(E):
    X, y = dset_array()
    X = np.reshape(X, (-1, 150**2))
    fri_idx = np.where(y < 4.5)
    frii_idx = np.where((y > 4.5) & (y < 7.5))
    hybrid_idx = np.where(y>7.5)

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(X)
    #embedding = StandardScaler().fit_transform(embedding)

    plt.scatter(embedding[fri_idx, 0], embedding[fri_idx, 1], c='red', label='fri')
    plt.scatter(embedding[frii_idx, 0], embedding[frii_idx, 1], c='blue', label='frii')
    plt.scatter(embedding[hybrid_idx, 0], embedding[hybrid_idx, 1], c='green', label='hybrid')
    plt.legend()
    plt.savefig(EVAL_PATH + '/umap.pdf')

def class_idx(y):
    fri_idx = np.where(y < 4.5)
    frii_idx = np.where((y > 4.5) & (y < 7.5))
    hybrid_idx = np.where(y>7.5)
    return fri_idx, frii_idx, hybrid_idx

def plot_z_real(E, epoch):
    X, y = dset_array()
    fri_idx, frii_idx, hybrid_idx = class_idx(y)
    embedding = E(torch.from_numpy(X).cuda())[0].cpu().detach().numpy()
    reducer = umap.UMAP()
    umap_embedding = reducer.fit_transform(X)   
    plt.scatter(umap_embedding[fri_idx, 0], embedding[fri_idx, 1], c='red', label='fri', s=2, marker = 'x')
    plt.scatter(umap_embedding[frii_idx, 0], embedding[frii_idx, 1], c='blue', label='frii', s=2, marker = 'x')
    plt.scatter(umap_embedding[hybrid_idx, 0], embedding[hybrid_idx, 1], c='green', label='hybrid', s=2, marker = 'x')
    plt.legend()
    plt.savefig(EMBEDDING_PATH + '/embedding_real_{}.pdf'.format(epoch))
    plt.close()

def plot_z_fake(X, E, epoch):
    embedding = E(X)[0].cpu().detach().numpy()
    reducer = umap.UMAP()
    umap_embedding = reducer.fit_transform(X)   
    plt.scatter(umap_embedding[:,  0], embedding[:, 1], c='red', s=2, marker = 'x')
    plt.savefig(EMBEDDING_PATH + '/embedding_fake_{}.pdf'.format(epoch))
    plt.close()

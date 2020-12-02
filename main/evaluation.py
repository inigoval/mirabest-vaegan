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

def renormalize(X, mu=0.0032, std = 0.0352):
    X = (X - mu)/std
    return X

"""def generate(G, n_z, n_samples=1000, n_batch=100):
    X = np.zeros((n_samples, 150, 150))
    for i in np.arange(n_samples/n_batch):
        Z_batch = torch.randn(n_batch, n_z).cuda().view(n_batch, n_z, 1, 1)
        X_batch = 
        X[i*n_batch:i+1*n_batch] = X_batch
    return X"""

def generate(G, n_z, n_samples=1000):
    Z = torch.randn(n_samples, n_z).cuda().view(n_samples, n_z, 1, 1)
    X = G(Z)
    return X

def inception_score(I, X, eps = 1E-10):
    # normalise X for CNN evaluation
    X = renormalize(X)
    p_yx = I(X)
    p_y = torch.mean(p_yx, 0)
    KL = torch.mean(p_yx * (torch.log(p_yx + eps) - torch.log(p_y + eps)))
    # squeeze inception score between 0 and 1
    #IS = KL/3
    IS = np.exp((KL-3))
    return IS

def frechet_distance(I, X_gen, X_real):
    X_gen, X_real = renormalize(X_gen), renormalize(X_real)
    f_gen, f_real = I(X_gen).fid_layer.detach().cpu().numpy(), I(X_real).fid_layer.detach().cpu().numpy()
    mu_gen, mu_real = np.mean(f_gen, axis=0), np.mean(f_real, axis=0)
    chi_gen, chi_real = np.cov(f_gen), np.cov(f_real)
    fid = np.mean((mu_gen-mu_real)^2, axis=0) + np.mean(np.trace((chi_gen + chi_real - 2*(np.matmul(chi_real, chi_gen))), axis1=1, axis2=2))
    return fid

def class_idx(y):
    fri_idx = np.where(y < 4.5)
    frii_idx = np.where((y > 4.5) & (y < 7.5))
    hybrid_idx = np.where(y>7.5)
    return fri_idx, frii_idx, hybrid_idx

def plot_eval_dict(eval_dict, epoch):
    fig, ax = plt.subplots(1,1)
    x_plot = eval_dict['x_plot']

    ## plot losses for each network ##
    ax.plot(x_plot[:epoch], eval_dict['inception'][:epoch], label='inception score')
    ax.set_xlabel('epoch')
    ax.legend()
    fig.savefig(EVAL_PATH + '/inception_score.pdf')
    plt.close(fig)

    fig, ax = plt.subplots(1,1)
    x_plot = eval_dict['x_plot']

    ## plot losses for each network ##
    ax.plot(x_plot[:epoch], eval_dict['frechet'][:epoch], label='frechet distance')
    ax.set_xlabel('epoch')
    ax.legend()
    fig.savefig(EVAL_PATH + '/frechet_distance.pdf')
    plt.close(fig)


def plot_z_real(X, y, E, epoch, n_z):
    fri_idx, frii_idx, hybrid_idx = class_idx(y)
    embedding = E(torch.from_numpy(X).cuda())[0].view(-1, n_z).cpu().detach().numpy()
    reducer = umap.UMAP()
    umap_embedding = reducer.fit_transform(embedding)   
    plt.scatter(umap_embedding[fri_idx, 0], embedding[fri_idx, 1], c='red', label='fri', s=2, marker = 'x')
    plt.scatter(umap_embedding[frii_idx, 0], embedding[frii_idx, 1], c='blue', label='frii', s=2, marker = 'x')
    plt.scatter(umap_embedding[hybrid_idx, 0], embedding[hybrid_idx, 1], c='green', label='hybrid', s=2, marker = 'x')
    plt.legend()
    plt.savefig(EMBEDDING_PATH + '/embedding_real_{}.pdf'.format(epoch))
    plt.close()

def plot_z_fake(X, E, epoch, n_z):
    embedding = E(X)[0].view(-1, n_z).cpu().detach().numpy()
    reducer = umap.UMAP()
    umap_embedding = reducer.fit_transform(embedding)   
    plt.scatter(umap_embedding[:,  0], embedding[:, 1], c='red', s=2, marker = 'x')
    plt.savefig(EMBEDDING_PATH + '/embedding_fake_{}.pdf'.format(epoch))
    plt.close()

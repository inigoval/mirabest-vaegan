import numpy as np
import umap
import torchvision
import torch
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.linalg import sqrtm

from dataloading import load_data, MiraBest_full
from utilities import y_collapsed, add_noise

# define paths for saving
FILE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EVAL_PATH = os.path.join(FILE_PATH, 'files', 'eval')
DATA_PATH = os.path.join(FILE_PATH, 'data')
CHECKPOINT_PATH = os.path.join(FILE_PATH, 'files', 'checkpoints')
FIG_PATH = os.path.join(FILE_PATH, 'files', 'figs')

EMBEDDING_PATH = os.path.join(EVAL_PATH, 'embeddings')

EMBEDDING_PATH_REAL = os.path.join(EMBEDDING_PATH, 'real')
EMBEDDING_PATH_FAKE = os.path.join(EMBEDDING_PATH, 'fake')

cuda = torch.device('cuda')


def dset_array(cuda=False):
    """
    DEPRECATED
    """
    ## load and normalise data ##
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0], std=[1])
    ])
    train_data = MiraBest_full(DATA_PATH, train=True, transform=transform, download=True)
    test_data = MiraBest_full(DATA_PATH, train=False, transform=transform, download=True)
    all_data = torch.utils.data.ConcatDataset((train_data, test_data))
    train_loader = torch.utils.data.DataLoader(all_data, batch_size=len(train_data), shuffle=True)
    X, y = next(iter(train_loader))
    if cuda:
        return X.cuda(), y.cuda()
    else:
        return X.cpu(), y.cpu()


def renormalize(X, mu=0.0031, std=0.0352):
    X = (X - mu)/std
    return X


def generate(G, n_z, n_samples=1000):
    Z = torch.randn((n_samples, n_z)).view(n_samples, n_z, 1, 1).cuda()
    X = G(Z)
    return X


def inception_score(I, X, eps=1E-10):
    # normalise X for CNN evaluation
    # X = renormalize(X)
    p_yx = I(X)
    p_y = torch.mean(p_yx, 0).view(1, -1)
    p_y = p_y.expand(X.size()[0], -1)
    KL = torch.mean(p_yx * torch.log(p_yx + eps) - torch.log(p_y + eps)).detach().cpu().numpy()
    # squeeze inception score between 0 and 1
    # IS = KL/3
    IS = np.exp((KL-2))
    return IS


def compute_mu_sig(I, data):
    """
    Compute the FID (hidden) layer mean and covariance for given data
    """
    _ = I(data)
    fid_layer = I.fid_layer.detach().cpu().numpy()
    mu = np.mean(fid_layer, axis=0)
    sigma = np.cov(fid_layer, rowvar=False)
    return mu, sigma


def fid(I, mu_real, sigma_real, X):
    # X_gen, X_real = renormalize(X_gen).cpu(), renormalize(X_real).cpu()
    mu_gen, sigma_gen = compute_mu_sig(I, X)

    S = sqrtm((np.dot(sigma_gen, sigma_real)))

    if np.iscomplexobj(S):
        S = S.real

    Dmu = np.square(mu_gen - mu_real).sum()

    fid = Dmu + np.trace((sigma_gen + sigma_real - 2*S), axis1=0, axis2=1)
    return fid


def test_prob(D, testLoader, n_test, bool_val, noise_scale, epoch, n_epochs):
    """ 
    Evaluate the discriminator output for held out real samples (to detect overfitting)
    A value close to 1 means close to no overfitting, a value close to zero implies
    significant overfitting
    """
    D_sum = 0.
    for data in testLoader:
        X, _ = data
        X = add_noise(bool_val, X.cuda(), noise_scale, epoch, n_epochs).cuda()
        D_X = D(X)[0].view(-1)
        D_sum += torch.sum(D_X).item()
    D_avg = D_sum/n_test
    return D_avg


def ratio(X, I):
    y = I(X)
    _, y_pred = torch.max(y, 1)
    n = y_pred.size()[0]
    n_fri, n_frii = (y_pred == 0).sum().item(), (y_pred == 1).sum().item()
    assert n_fri + n_frii == n
    R = n_fri/n
    return R*100


def class_idx(y):
    fri_idx = np.argwhere(y == 0)
    frii_idx = np.argwhere(y == 1)
    hybrid_idx = np.argwhere(y == 2)
    return fri_idx, frii_idx, hybrid_idx


def plot_eval_dict(eval_dict, epoch):
    x_plot = eval_dict['n_samples']
    epsilon = eval_dict['epsilon']
    FID, D_X_test, R = eval_dict['fid'], eval_dict['D_X_test'], eval_dict['fri%']

    ## plot frechet inception distance ##
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('FID')
    ax1.plot(x_plot[:epoch], FID[:epoch], label='frechet distance')
    ax1.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    ax1.set_ylim(0, 500)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('epsilon', color=color)
    ax2.plot(x_plot[:epoch], epsilon[:epoch], '--')
    ax2.tick_params(axis='y', labelcolor=color)
    fig.savefig(EVAL_PATH + '/frechet_distance.pdf')
    plt.close(fig)

    ## plot frechet inception distance ##
    fig, ax = plt.subplots(1, 1)
    ax.plot(x_plot[:epoch], FID[:epoch], label='frechet distance')
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    ax.set_xlabel('epoch')
    ax.set_ylabel('FID')
    ax.set_ylim(0, 150)
    fig.savefig(EVAL_PATH + '/frechet_distance-zoomed.pdf')
    plt.close(fig)

    ## plot frechet inception distance ##
    fig, ax = plt.subplots(1, 1)
    ax.plot(x_plot[:epoch], FID[:epoch], label='frechet distance')
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    ax.set_xlabel('epoch')
    ax.set_ylabel('FID')
    ax.set_ylim(0, 50)
    fig.savefig(EVAL_PATH + '/frechet_distance-extra-zoomed.pdf')
    plt.close(fig)

    ## plot overfitting score, 1 is no overfitting 0 is completely overfitted ##
    fig, ax = plt.subplots(1, 1)
    ax.plot(x_plot[:epoch], D_X_test[:epoch], label='D(X_test)')
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    ax.set_xlabel('epoch')
    ax.set_ylabel('D(X_test)')
    ax.legend()
    ax.set_ylim(0, 1)
    fig.savefig(EVAL_PATH + '/overfitting_score.pdf')
    plt.close(fig)

    ## plot fri% to assess bias of generator ##
    '''fig, ax = plt.subplots(1, 1)
    ax.plot(x_plot[:epoch], R[:epoch], label='fri%')
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    ax.set_xlabel('epoch')
    ax.set_ylabel('fri%')
    ax.legend()
    ax.set_ylim(0, 100)
    fig.savefig(EVAL_PATH + '/fri%.pdf')
    plt.close(fig)'''


def plot_z_real(X, y, E, epoch, n_z):
    with torch.no_grad():
        fri_idx, frii_idx, hybrid_idx = class_idx(y.numpy())
        embedding = E(X.cuda())[0].view(-1, n_z).cpu().detach().numpy()
        reducer = umap.UMAP()
        umap_embedding = reducer.fit_transform(embedding)
        plt.scatter(umap_embedding[fri_idx, 0], embedding[fri_idx, 1], c='red', label='fri', s=2, marker='x')
        plt.scatter(umap_embedding[frii_idx, 0], embedding[frii_idx, 1], c='blue', label='frii', s=2, marker='x')
        plt.scatter(umap_embedding[hybrid_idx, 0], embedding[hybrid_idx, 1], c='green', label='hybrid', s=2, marker='x')
        plt.legend()
        plt.savefig(EMBEDDING_PATH_REAL + '/embedding_real_{}.pdf'.format(epoch))
        plt.close()


def plot_z_fake(I, X, E, epoch, n_z):
    with torch.no_grad():
        embedding = E(X)[0].view(-1, n_z).cpu().detach().numpy()
        y_hat = I(X.cpu())
        _, y_pred = torch.max(y_hat, 1)
        y_pred = y_pred.numpy()
        fri_idx, frii_idx, hybrid_idx = class_idx(y_pred)
        reducer = umap.UMAP()
        umap_embedding = reducer.fit_transform(embedding)
        plt.scatter(umap_embedding[fri_idx, 0], embedding[fri_idx, 1], c='red', label='fri', s=2, marker='x')
        plt.scatter(umap_embedding[frii_idx, 0], embedding[frii_idx, 1], c='blue', label='frii', s=2, marker='x')
        plt.scatter(umap_embedding[hybrid_idx, 0], embedding[hybrid_idx, 1], c='green', label='hybrid', s=2, marker='x')
        plt.legend()
        plt.savefig(EMBEDDING_PATH_FAKE + '/embedding_fake_{}.pdf'.format(epoch))
        plt.close()


def plot_z(X, y, E, epoch):
    X, y = X.detach().numpy(), y.detach().numpy()
    fri_idx, frii_idx, hybrid_idx = class_idx(y)
    embedding = E(torch.from_numpy(X).cuda())[0].cpu().detach().numpy()
    plt.scatter(embedding[fri_idx, 0], embedding[fri_idx, 1], c='red', label='fri')
    plt.scatter(embedding[frii_idx, 0], embedding[frii_idx, 1], c='blue', label='frii')
    plt.scatter(embedding[hybrid_idx, 0], embedding[hybrid_idx, 1], c='green', label='hybrid')
    plt.legend()
    plt.savefig(FIG_PATH + '/embedding_{}.pdf'.format(epoch))
    plt.close()

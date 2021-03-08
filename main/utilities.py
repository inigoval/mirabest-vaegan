import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import yaml

from mpl_toolkits.axes_grid1 import ImageGrid
from dataloading import MiraBest_full

# define paths for saving
# define paths for saving
FILE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EVAL_PATH = os.path.join(FILE_PATH, 'files', 'eval')
DATA_PATH = os.path.join(FILE_PATH, 'data')
CFG_PATH = os.path.join(FILE_PATH, 'configs')
CHECKPOINT_PATH = os.path.join(FILE_PATH, 'files', 'checkpoints')
FIG_PATH = os.path.join(FILE_PATH, 'files', 'figs')
IMAGE_PATH = os.path.join(FILE_PATH, 'files', 'images')
CFG_PATH = os.path.join(FILE_PATH, 'config')

EMBEDDING_PATH = os.path.join(EVAL_PATH, 'embeddings')
FAKE_PATH = os.path.join(IMAGE_PATH, 'fake')
RECON_PATH = os.path.join(IMAGE_PATH, 'reconstructed')

class Annealed_Noise():
    """
    Adds noise of a given amplitude to the input images X
    Leave the last quarter of epochs to optimise without any noise
    """
    def __init__(self, noise_scale, n_epochs):
        self.n_epochs = n_epochs
        self.noise_scale = noise_scale
        self.epsilon = 0

    def __call__(self, X):
        if self.noise_scale > 0:
            noise = torch.randn_like(X)*self.epsilon*self.noise_scale
            X = X + noise  
            return X
        else:
            return X

    def update_epsilon(self, epoch):
        epsilon = 1 - 1.33333333*epoch/self.n_epochs
        epsilon = np.clip(epsilon, 0, 1)
        self.epsilon = epsilon


class Plot_Images():
    """
    Class designed to plot image grids with the same latent vector/real image input
    """
    def __init__(self, X_full, n_z, grid_length=6, recon_path=RECON_PATH, fake_path=FAKE_PATH):
        self.grid_length = grid_length
        idx = np.random.randint(0, X_full.shape[0], int((grid_length**2)/2))
        self.X = X_full[idx, ...].cuda()
        self.Z = torch.randn(grid_length**2, n_z).cuda().view(grid_length**2, n_z, 1, 1)
        self.recon_path = recon_path
        self.fake_path = fake_path
        self.H = X_full.shape[-1]
        self.W = X_full.shape[-2]

    def plot(self, path):
        img_list = list(self.img_array)
        assert self.grid_length == int(len(img_list)**0.5)
        fig = plt.figure(figsize=(13., 13.))
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(self.grid_length, self.grid_length),
                         axes_pad=0)

        for ax, im in zip(grid, img_list):
            im = im.reshape((self.H, self.W))
            ax.axis('off')
            ax.imshow(im, cmap='hot')
        plt.axis('off')
        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)

    def generate(self, E, G, recon=False):
        if recon:
            mu, logvar = E(self.X)
            X_recon = G(mu).detach().cpu().numpy()
            X = np.concatenate((X_recon, self.X.detach().cpu().numpy()), axis=0)
        else:
            X = G(self.Z).view(-1, self.H, self.W).detach().cpu().numpy()
        
        self.img_array = X

    def plot_generate(self, E, G, filename='images.pdf', recon=False):
        self.generate(E, G, recon=recon)
        if recon:
            path = os.path.join(self.recon_path, filename)
        else:
            path = os.path.join(self.fake_path, filename)
        self.plot(path)


def load_config(config_path=CFG_PATH):
    """
    Helper function to load config file
    """
    path = os.path.join(config_path, 'config.yml')
    with open(path, "r") as ymlcfg:
        cfg = yaml.load(ymlcfg, Loader=yaml.FullLoader)
    return cfg


def eps_noise(epoch, n_epochs):
    epsilon = torch.clamp(torch.Tensor([1 - 1.33333333*epoch/n_epochs]), min=0, max=1)
    return epsilon

    
def add_noise(bool_val, X, noise_scale, epsilon):
    """
    Adds noise of a given amplitude to the input images X
    Leave the last quarter of epochs to optimise without any noise
    """
    def __init__(self, noise_scale):
        self.noise_scale = noise_scale
        self.epsilon = 0

    def __call__(self, X):
        if self.noise_scale > 0:
            noise = torch.randn_like(X)*self.epsilon*self.noise_scale
            X = X + noise  
            return X
        else:
            return X

    def update_epsilon(self, epoch, n_epochs):
        epsilon = 1 - 1.33333333*epoch/n_epochs
        epsilon = np.clip(epsilon, 0, 1)
        self.epsilon = epsilon


class Plot_Images():
    def __init__(self, X_full, n_z, grid_length=6, recon_path=RECON_PATH, fake_path=FAKE_PATH):
        self.grid_length = grid_length
        idx = np.random.randint(0, X_full.shape[0], int((grid_length**2)/2))
        self.X = X_full[idx, ...].cuda()
        self.Z = torch.randn(grid_length**2, n_z).cuda().view(grid_length**2, n_z, 1, 1)
        self.recon_path = recon_path
        self.fake_path = fake_path
        self.H = X_full.shape[-1]
        self.W = X_full.shape[-2]

    def plot(self, path):
        img_list = list(self.img_array)
        assert self.grid_length == int(len(img_list)**0.5)
        fig = plt.figure(figsize=(13., 13.))
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(self.grid_length, self.grid_length),
                         axes_pad=0)

        for ax, im in zip(grid, img_list):
            im = im.reshape((self.H, self.W))
            ax.axis('off')
            ax.imshow(im, cmap='hot')
        plt.axis('off')
        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)

    def generate(self, E, G, recon=False):
        if recon:
            mu, logvar = E(self.X)
            X_recon = G(mu).detach().cpu().numpy()
            X = np.concatenate((X_recon, self.X.detach().cpu().numpy()), axis=0)
        else:
            X = G(self.Z).view(-1, self.H, self.W).detach().cpu().numpy()
        
        self.img_array = X

    def plot_generate(self, E, G, filename='images.pdf', recon=False):
        self.generate(E, G, recon=recon)
        if recon:
            path = os.path.join(self.recon_path, filename)
        else:
            path = os.path.join(self.fake_path, filename)
        self.plot(path)


class Labels():
    def __init__(self, p_flip):
        self.p_flip = p_flip

    def zeros(self, n):
        zeros = torch.zeros(n, dtype=torch.float).cuda()
        if self.p_flip > 0:
            n_flip = int(self.p_flip*n)
            flip_idx = torch.randint(n, (n_flip,))
            zeros[flip_idx] = 0
        return zeros
 
    def ones(self, n):
        ones = torch.ones(n, dtype=torch.float).cuda()
        if self.p_flip > 0:
            n_flip = int(self.p_flip*n)
            flip_idx = torch.randint(n, (n_flip,))
            ones[flip_idx] = 0   
        return ones 

    @staticmethod
    def p_flip_ann(epoch, n_epochs):
        p_flip = 1 - torch.FloatTensor(np.array(epoch/(n_epochs*0.5)))
        p_flip = torch.clamp(p_flip, 0, 1)
        return p_flip


class Set_Model():
    def __init__(self):
        pass
    
    @staticmethod
    def train(*models):
        for model in models:
            model.train()

    @staticmethod
    def eval(*models):
        for model in models:
            model.eval()

    @staticmethod
    def requires_grad(bool_val, *models):
        for model in models:
            for p in model.parameters():
                p.requires_grad = bool_val

    @staticmethod
    def weights_init(*models):
        for m in models:
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)


def load_config(config_path=CFG_PATH):
    """
    Helper function to load config file
    """
    path = os.path.join(config_path, 'config.yml')
    with open(path, "r") as ymlcfg:
        cfg = yaml.load(ymlcfg, Loader=yaml.FullLoader)
    return cfg


def KL_loss(mu, logvar):
    kl = -0.5 * torch.mean(1 + logvar - logvar.exp() - mu.pow(2)).view(1)
    #print('kl loss: ', kl)
    assert kl.size() ==torch.ones(1).size()
    return kl


def z_sample(mu, logvar):
    std = logvar.exp().pow(0.5)
    epsilon = torch.randn_like(std)  # returns tensor same size as std but mean 0 var 1
    z_tilde = torch.add(mu, torch.mul(epsilon, std))
    assert z_tilde.size() == mu.size()
    return z_tilde


def plot_losses(L_dict, epoch):
    x_plot = L_dict['x_plot']

    ## plot losses for each network ##
    fig, ax = plt.subplots(1, 1)
    ax.plot(x_plot[:epoch], L_dict['L_E'][:epoch], label='encoder loss')
    ax.plot(x_plot[:epoch], L_dict['L_G'][:epoch], label='decoder loss')
    ax.plot(x_plot[:epoch], L_dict['L_D'][:epoch], label='discriminator loss')
    ax.set_xlabel('epoch')
    ax.legend()
    fig.savefig(FIG_PATH + '/network_losses.pdf')
    plt.close(fig)

    ## plot average predicted probabilities for generated/reconstructed images ##
    fig, ax = plt.subplots(1, 1)
    x_plot = L_dict['x_plot']
    ax.plot(x_plot[:epoch], L_dict['y_gen'][:epoch], label='D(G(z))')
    ax.plot(x_plot[:epoch], L_dict['y_recon'][:epoch], label='D(G(E(X)))')
    #ax.plot(x_plot[:epoch], half[:epoch], 'g:', label = 'p(real) = 0.5')
    ax.set_xlabel('epoch')
    ax.legend()
    fig.savefig(FIG_PATH + '/disc_predictions.pdf')
    plt.close(fig)


def plot_images(X, E, G, n_z, epoch):
    """
    Plot and save images from both randomly generated z vectors and
    z vectors encoded from real data
    """
    # generate random z values
    Z_plot = torch.normal(torch.zeros(4, n_z), torch.ones(4, n_z)).cuda().view(4, n_z, 1, 1)
    X_gen = G(Z_plot).cpu().detach().view(-1, 150, 150).numpy()
    # set up a 2x2 grid of plots in figure 2
    # [ax21  ax22]
    # [ax23  ax24]
    fig, ((ax21, ax22), (ax23, ax24)) = plt.subplots(2, 2)
    im1 = ax21.imshow(X_gen[0, :, :], cmap='hot')
    im2 = ax22.imshow(X_gen[1, :, :], cmap='hot')
    im3 = ax23.imshow(X_gen[2, :, :], cmap='hot')
    im4 = ax24.imshow(X_gen[3, :, :], cmap='hot')
    fig.savefig(FAKE_PATH + f'/epoch_{epoch+1}.pdf')
    plt.close(fig)

    X_sample = X[:2, :, :, :]
    X_recon = G(E(X_sample)[0]).cpu().detach().view(-1, 150, 150).numpy()
    # set up a 2x2 grid of plots in figure 2
    # [ax21  ax22]
    # [ax23  ax24]
    fig, ((ax21, ax22), (ax23, ax24)) = plt.subplots(2, 2)
    im1 = ax21.imshow(X_recon[0, :, :], cmap='hot')
    im2 = ax22.imshow(X_recon[1, :, :], cmap='hot')
    im3 = ax23.imshow(X_sample[0, :, :].cpu().view(150, 150).numpy(), cmap='hot')
    im4 = ax24.imshow(X_sample[1, :, :].cpu().view(150, 150).numpy(), cmap='hot')
    fig.savefig(RECON_PATH + f'/recon_epoch_{epoch+1}.pdf')
    plt.close(fig)

import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import yaml
from pathlib import Path

from mpl_toolkits.axes_grid1 import ImageGrid

from paths import Path_Handler

# define paths for saving
paths = Path_Handler()
path_dict = paths._dict()


class Annealed_Noise:
    """
    Adds noise of a given amplitude to the input images X
    Leave the last quarter of epochs to optimise without any noise
    """

    def __init__(self, noise_scale, n_epochs):
        self.n_epochs = n_epochs
        self.noise_scale = noise_scale
        self.epsilon = 0
        self.noise = 0

    def __call__(self, X, update_noise=True):
        if self.noise_scale > 0:
            if update_noise:
                self.noise = torch.randn_like(X) * self.epsilon * self.noise_scale

            X = X + self.noise
            return X
        else:
            return X

    def update_epsilon(self, epoch):
        epsilon = 1 - 1.33333333 * epoch / self.n_epochs
        epsilon = np.clip(epsilon, 0, 1)
        self.epsilon = epsilon


class Plot_GAug:
    def __init__(self, X_full, path_dict, grid_length=4):
        self.grid_length = grid_length
        self.X_full = X_full
        self.path_dict = path_dict
        self.H = X_full.shape[-1]
        self.W = X_full.shape[-2]

    def GAug(self, E, G, alpha=0, idx=0):
        self.alpha = alpha
        self.idx = idx
        img = self.X_full[idx, ...].cuda()
        n_gen = self.grid_length ** 2 - 1
        mu, logvar = E(img.view(1, 1, self.H, self.W))
        mu = mu.expand(n_gen, -1, -1, -1)
        logvar = logvar.expand(n_gen, -1, -1, -1)
        std = alpha * torch.exp(0.5 * logvar)
        Z = std * torch.randn_like(mu) + mu
        X = G(Z).view(-1, self.H, self.W).detach().cpu().numpy()
        X = np.concatenate((img.detach().cpu().numpy(), X), axis=0)
        self.img_array = X

    def plot(self, epoch):
        path = (
            self.path_dict["gaug"]
            / f"epoch{epoch+1}_img{self.idx}_alpha{self.alpha}.png"
        )
        img_list = list(self.img_array)
        assert self.grid_length == int(len(img_list) ** 0.5)
        fig = plt.figure(figsize=(13.0, 13.0))
        grid = ImageGrid(
            fig, 111, nrows_ncols=(self.grid_length, self.grid_length), axes_pad=0
        )

        for ax, im in zip(grid, img_list):
            im = im.reshape((self.H, self.W))
            ax.axis("off")
            ax.imshow(im, cmap="hot")
        plt.axis("off")
        plt.savefig(path, bbox_inches="tight")
        plt.close(fig)


class Plot_Images:
    """
    Class designed to plot image grids with the same latent vector/real image input
    """

    def __init__(self, X_full, n_z, path_dict, grid_length=6, alpha=0):
        self.grid_length = grid_length
        idx = np.random.randint(0, X_full.shape[0], int((grid_length ** 2) / 2))
        self.X = X_full[idx, ...].cuda()
        self.Z = (
            torch.randn(grid_length ** 2, n_z).cuda().view(grid_length ** 2, n_z, 1, 1)
        )
        self.recon_path = path_dict["recon"]
        self.fake_path = path_dict["fake"]
        self.H = X_full.shape[-1]
        self.W = X_full.shape[-2]

    def plot(self, path):
        img_list = list(self.img_array)
        assert self.grid_length == int(len(img_list) ** 0.5)
        fig = plt.figure(figsize=(13.0, 13.0))
        grid = ImageGrid(
            fig, 111, nrows_ncols=(self.grid_length, self.grid_length), axes_pad=0
        )

        for ax, im in zip(grid, img_list):
            im = im.reshape((self.H, self.W))
            ax.axis("off")
            ax.imshow(im, cmap="hot")
        plt.axis("off")
        plt.savefig(path, bbox_inches="tight")
        plt.close(fig)

    def generate(self, E, G, recon=False):
        if recon:
            mu, logvar = E(self.X)
            X_recon = G(mu).detach().cpu().numpy()
            X = np.concatenate((X_recon, self.X.detach().cpu().numpy()), axis=0)
        else:
            X = G(self.Z).view(-1, self.H, self.W).detach().cpu().numpy()

        self.img_array = X

    def plot_generate(self, E, G, filename="images.pdf", recon=False):
        self.generate(E, G, recon=recon)
        if recon:
            path = os.path.join(self.recon_path, filename)
        else:
            path = os.path.join(self.fake_path, filename)
        self.plot(path)


class Labels:
    def __init__(self, p_flip):
        self.p_flip = p_flip

    def zeros(self, n):
        zeros = torch.zeros(n, dtype=torch.float).cuda()
        if self.p_flip > 0:
            n_flip = int(self.p_flip * n)
            flip_idx = torch.randint(n, (n_flip,))
            zeros[flip_idx] = 1
        return zeros

    def ones(self, n):
        ones = torch.ones(n, dtype=torch.float).cuda()
        if self.p_flip > 0:
            n_flip = int(self.p_flip * n)
            flip_idx = torch.randint(n, (n_flip,))
            ones[flip_idx] = 0
        return ones

    @staticmethod
    def p_flip_ann(epoch, n_epochs):
        p_flip = 1 - torch.FloatTensor(np.array(epoch / (n_epochs * 0.5)))
        p_flip = torch.clamp(p_flip, 0, 1)
        return p_flip


def load_config(config_path=path_dict["config"]):
    """
    Helper function to load config file
    """
    path = os.path.join(config_path, "config.yml")
    with open(path, "r") as ymlcfg:
        cfg = yaml.load(ymlcfg, Loader=yaml.FullLoader)
    return cfg


def KL_loss(mu, logvar):
    kl = -0.5 * torch.mean(1 + logvar - logvar.exp() - mu.pow(2)).view(1)
    # print('kl loss: ', kl)
    assert kl.size() == torch.ones(1).size()
    return kl


def z_sample(mu, logvar):
    std = logvar.exp().pow(0.5)
    epsilon = torch.randn_like(std)  # returns tensor same size as std but mean 0 var 1
    z_tilde = torch.add(mu, torch.mul(epsilon, std))
    assert z_tilde.size() == mu.size()
    return z_tilde


class Set_Model:
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
            if classname.find("Conv") != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find("BatchNorm") != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)


def plot_hist(E, X, epoch, eval=False):
    if not eval:
        E.train()
    mu, logvar = E(X.cuda())
    mu = mu.view(-1).detach().cpu().numpy()
    std = torch.exp(0.5 * logvar)
    std = std.view(-1).detach().cpu().numpy()
    plt.hist(mu, bins="auto")
    plt.savefig(path_dict["hist"] / f"mu_hist_e{epoch+1}{eval}")
    plt.hist(std, bins="auto")
    plt.savefig(path_dict["hist"] / f"mu_hist_e{epoch+1}{eval}")
    plt.close()
    E.eval()


def plot_losses(L_dict, epoch):
    x_plot = L_dict["x_plot"]

    ## plot losses for each network ##
    fig, ax = plt.subplots(1, 1)
    ax.plot(x_plot[:epoch], L_dict["L_E"][:epoch], label="encoder loss")
    ax.plot(x_plot[:epoch], L_dict["L_G"][:epoch], label="decoder loss")
    ax.plot(x_plot[:epoch], L_dict["L_D"][:epoch], label="discriminator loss")
    ax.set_xlabel("epoch")
    ax.legend()
    fig.savefig(path_dict["eval"] / "network_losses.pdf")
    plt.close(fig)

    ## plot average predicted probabilities for generated/reconstructed images ##
    fig, ax = plt.subplots(1, 1)
    x_plot = L_dict["x_plot"]
    ax.plot(x_plot[:epoch], L_dict["y_gen"][:epoch], label="D(G(z))")
    ax.plot(x_plot[:epoch], L_dict["y_recon"][:epoch], label="D(G(E(X)))")
    # ax.plot(x_plot[:epoch], half[:epoch], 'g:', label = 'p(real) = 0.5')
    ax.set_xlabel("epoch")
    ax.legend()
    fig.savefig(FIG_PATH + "/disc_predictions.pdf")
    plt.close(fig)

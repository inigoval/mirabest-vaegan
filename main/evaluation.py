import numpy as np
import umap
import torchvision
import torch
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.linalg import sqrtm

from paths import Path_Handler

paths = Path_Handler()
path_dict = paths._dict()


# define paths for saving

cuda = torch.device("cuda")


class FID:
    """
    Initialise the class with classifier model and training data to use as reference
    """

    def __init__(self, I, data):
        self.I = I
        self.data = data
        self.mu_real, self.sigma_real = self.compute_mu_sig(I, data)

    def __call__(self, X_fake):
        mu, sigma = self.compute_mu_sig(self.I, X_fake)

        S = sqrtm((np.dot(sigma, self.sigma_real)))

        if np.iscomplexobj(S):
            S = S.real

        Dmu = np.square(mu - self.mu_real).sum()

        fid = Dmu + np.trace((sigma + self.sigma_real - 2 * S), axis1=0, axis2=1)
        self.fid = fid
        return fid

    def reconstruct(data, E, G, n_samples=1000):
        Z, _ = E(data[:n_samples, ...].cuda())
        X = G(Z)
        return X

    @staticmethod
    def compute_mu_sig(I, data):
        _ = I(data)
        fid_layer = I.fid_layer.detach().cpu().numpy()
        mu = np.mean(fid_layer, axis=0)
        sigma = np.cov(fid_layer, rowvar=False)
        return mu, sigma

    @staticmethod
    def generate(G, n_z, n_samples=1000):
        Z = torch.randn((n_samples, n_z)).view(n_samples, n_z, 1, 1).cuda()
        X = G(Z)
        return X


class Eval:
    def __init__(self, n_epochs):
        self.epochs = np.arange(n_epochs)
        self.samples = np.zeros(n_epochs)
        self.fid = {"fake": np.zeros(n_epochs), "recon": np.zeros(n_epochs)}
        self.D_X_test = np.zeros(n_epochs)
        self.ratio = np.zeros(n_epochs)
        self.epsilon = np.zeros(n_epochs)
        self.epoch = 0

    def update_epoch(self, epoch):
        self.epoch = epoch

    def calc_overfitscore(self, D, testLoader, n_test, noise):
        """
        Evaluate the discriminator output for held out real samples (to detect overfitting)
        A value close to 1 means close to no overfitting, a value close to zero implies
        significant overfitting
        """
        D_sum = 0.0
        for data in testLoader:
            X, _ = data
            X = noise(X.cuda()).cuda()
            D_X = D(X)[0].view(-1)
            D_sum += torch.sum(D_X).item()
        D_avg = D_sum / n_test
        self.D_X_test[self.epoch] = D_avg

    def calc_ratio(self, X, I):
        y = I(X)
        _, y_pred = torch.max(y, 1)
        n = y_pred.size()[0]
        n_fri, n_frii = (y_pred == 0).sum().item(), (y_pred == 1).sum().item()
        assert n_fri + n_frii == n
        R = n_fri / n
        self.ratio[self.epoch] = R * 100

    def plot_fid(self, eps=False, ylim=400, fid_type="fake"):
        x = self.epochs
        fid = self.fid[fid_type]
        epoch = self.epoch
        epsilon = self.epsilon

        ## plot frechet inception distance ##
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("n")
        ax1.set_ylabel("FID")
        ax1.plot(x[:epoch], fid[:epoch], label="frechet distance")
        ax1.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        # ax1.set_xticklabels([f'{t:.3e}' for t in ax1.get_yticks()])
        ax1.set_ylim(0, ylim)

        if eps:
            ax2 = ax1.twinx()
            color = "tab:red"
            ax2.set_ylabel("epsilon", color=color)
            ax2.plot(x[:epoch], epsilon[:epoch], "--", color="red")
            ax2.tick_params(axis="y", labelcolor=color)

        fig.savefig(path_dict["eval"] / f"fid_{fid_type}_{ylim}.pdf")
        plt.close(fig)

    def plot_overfitscore(self):
        x = self.epochs
        epoch = self.epoch
        D_X_test = self.D_X_test

        fig, ax = plt.subplots(1, 1)
        ax.plot(x[:epoch], D_X_test[:epoch], label="D(X_test)")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        # ax.set_xticklabels([f'{t:.3e}' for t in ax.get_yticks()])
        ax.set_xlabel("n")
        ax.set_ylabel("D(X_test)")
        ax.legend()
        ax.set_ylim(0, 1)
        fig.savefig(path_dict["eval"] / "overfitting_score.pdf")
        plt.close(fig)


def renormalize(X, mu=0.0031, std=0.0352):
    X = (X - mu) / std
    return X


def class_idx(y):
    fri_idx = np.argwhere(y == 0)
    frii_idx = np.argwhere(y == 1)
    hybrid_idx = np.argwhere(y == 2)
    return fri_idx, frii_idx, hybrid_idx


def plot_z_real(X, y, E, epoch, n_z):
    with torch.no_grad():
        fri_idx, frii_idx, hybrid_idx = class_idx(y.numpy())
        embedding = E(X.cuda())[0].view(-1, n_z).cpu().detach().numpy()
        reducer = umap.UMAP()
        umap_embedding = reducer.fit_transform(embedding)
        plt.scatter(
            umap_embedding[fri_idx, 0],
            embedding[fri_idx, 1],
            c="red",
            label="fri",
            s=2,
            marker="x",
        )
        plt.scatter(
            umap_embedding[frii_idx, 0],
            embedding[frii_idx, 1],
            c="blue",
            label="frii",
            s=2,
            marker="x",
        )
        plt.scatter(
            umap_embedding[hybrid_idx, 0],
            embedding[hybrid_idx, 1],
            c="green",
            label="hybrid",
            s=2,
            marker="x",
        )
        plt.legend()
        plt.savefig(EMBEDDING_PATH_REAL + "/embedding_real_{}.pdf".format(epoch))
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
        plt.scatter(
            umap_embedding[fri_idx, 0],
            embedding[fri_idx, 1],
            c="red",
            label="fri",
            s=2,
            marker="x",
        )
        plt.scatter(
            umap_embedding[frii_idx, 0],
            embedding[frii_idx, 1],
            c="blue",
            label="frii",
            s=2,
            marker="x",
        )
        plt.scatter(
            umap_embedding[hybrid_idx, 0],
            embedding[hybrid_idx, 1],
            c="green",
            label="hybrid",
            s=2,
            marker="x",
        )
        plt.legend()
        plt.savefig(EMBEDDING_PATH_FAKE + "/embedding_fake_{}.pdf".format(epoch))
        plt.close()


def plot_z(X, y, E, epoch):
    X, y = X.detach().numpy(), y.detach().numpy()
    fri_idx, frii_idx, hybrid_idx = class_idx(y)
    embedding = E(torch.from_numpy(X).cuda())[0].cpu().detach().numpy()
    plt.scatter(embedding[fri_idx, 0], embedding[fri_idx, 1], c="red", label="fri")
    plt.scatter(embedding[frii_idx, 0], embedding[frii_idx, 1], c="blue", label="frii")
    plt.scatter(
        embedding[hybrid_idx, 0], embedding[hybrid_idx, 1], c="green", label="hybrid"
    )
    plt.legend()
    plt.savefig(FIG_PATH + "/embedding_{}.pdf".format(epoch))
    plt.close()

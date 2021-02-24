import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import ast
import configparser
from mpl_toolkits.axes_grid1 import ImageGrid
from dataloading import MiraBest_full

# define paths for saving
# define paths for saving
FILE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EVAL_PATH = os.path.join(FILE_PATH, 'files', 'eval')
DATA_PATH = os.path.join(FILE_PATH, 'data')
CHECKPOINT_PATH = os.path.join(FILE_PATH, 'files', 'checkpoints')
FIG_PATH = os.path.join(FILE_PATH, 'files', 'figs')
IMAGE_PATH = os.path.join(FILE_PATH, 'files', 'images')

EMBEDDING_PATH = os.path.join(EVAL_PATH, 'embeddings')
FAKE_PATH = os.path.join(IMAGE_PATH, 'fake')
RECON_PATH = os.path.join(IMAGE_PATH, 'reconstructed')

class add_noise():
    """
    Adds noise of a given amplitude to the input images X
    Leave the last quarter of epochs to optimise without any noise
    """
    def __init__(self, noise_scale, epoch, n_epochs):
        self.noise_scale = noise_scale
        self.epoch = epoch
        self.n_epochs = n_epochs
        self.epsilon = self.get_epsilon(epoch, n_epochs)

    def __call__(self, X):
        if self.noise_scale > 0:
            noise = torch.randn_like(X)*self.epsilon*self.noise_scale
            X_noisey += noise  
            return X_noisey
        else:
            return X

    @staticmethod
    def get_epsilon(epoch, n_epochs):
        epsilon = torch.clamp(torch.Tensor([1 - 1.33333333*epoch/n_epochs]), min=0, max=1)
        return epsilon


def p_flip_ann(epoch, n_epochs):
    p_flip = 1 - torch.FloatTensor(np.array(epoch/(n_epochs*0.5)))
    p_flip = torch.clamp(p_flip, 0, 1)
    return p_flip


def labels(label_flip, smoothing, p_flip, smoothing_scale, n_X):
    ones = torch.ones(n_X, dtype=torch.float).cuda()
    zeros = torch.zeros(n_X, dtype=torch.float).cuda()

    ## flip labels ##
    if label_flip == True:
        n_flip = int(p_flip*n_X)
        flip_idx = torch.randint(n_X, (n_flip,))
        ones[flip_idx] = 0
        flip_idx = torch.randint(n_X, (n_flip,))
        zeros[flip_idx] = 1

    ## smooth labels ##
    if smoothing == True:
        ones = ones - np.abs(np.random.normal(loc=0.0, scale=smoothing_scale))

    return ones, zeros


def set_train(*models):
    for model in models:
        model.train()


def KL_loss(mu, logvar):
    kl = -0.5 * torch.mean(1 + logvar - logvar.exp() - mu.pow(2)).view(1)
    #print('kl loss: ', kl)
    assert kl.size() ==torch.ones(1).size()
    return kl


def z_sample(mu, logvar):
    std = logvar.exp().pow(0.5)
    epsilon = torch.randn_like(std)  # returns tensor same size as std but mean 0 var 1
    z_tilde = torch.add(mu, torch.mul(epsilon, std))
    # print(z_tilde.requires_grad)
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


def plot_grid(n_z, E, G, Z_plot, epoch, n_images=6):
    img_list_p = []
    img_list_tilde = []
    img_list_data = []

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    data = MiraBest_full(DATA_PATH, train=True, transform=transform, download=True)

    Z_plot = torch.randn(n_images**2, n_z).cuda().view(n_images**2, n_z, 1, 1)

    ## list of randomly generated images ##
    for i in range(n_images**2):
        z = Z_plot[i].view(1, n_z, 1, 1)
        X_p = G(z).cpu().detach().view(150, 150)
        img_list_p.append(X_p)

    ## generate reconstructed image list ##
    for i in range(int(n_images**2/2)):
        X = data.__getitem__(i)[0].cuda().view(1, 1, 150, 150)
        z = E(X)[0].view(1, n_z, 1, 1)
        X_tilde = G(z).cpu().detach().view(150, 150)
        img_list_tilde.append(X_tilde.cpu().detach().view(150, 150))
        img_list_data.append(X.cpu().detach().view(150, 150))

    img_list_joined = img_list_tilde + img_list_data

    fig = plt.figure(figsize=(13., 13.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(n_images, n_images),  # creates 2x2 grid of axes
                     axes_pad=0,  # pad between axes in inch.
                     )

    for ax, im in zip(grid, img_list_p):
        ax.imshow(im, cmap='hot')
    #plt.title('generated images epoch {}'.format(epoch))
    plt.savefig(FAKE_PATH + f'/grid_X_p_{epoch+1}.pdf')
    plt.close(fig)

    fig = plt.figure(figsize=(13., 13.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(n_images, n_images),  # creates 2x2 grid of axes
                     axes_pad=0,  # pad between axes in inch.
                     )

    for ax, im in zip(grid, img_list_joined):
        ax.imshow(im, cmap='hot')
    plt.savefig(RECON_PATH + f'/grid_X_tilde_{epoch+1}.pdf')
    plt.close(fig)


def y_collapsed(y):
    """
    DEPRECATED
    """
    fri = torch.full((y.shape[0],), 0, dtype=torch.long)
    frii = torch.full((y.shape[0],), 1, dtype=torch.long)
    hybrid = torch.full((y.shape[0],), 2, dtype=torch.long)
    y = torch.where(y > 4.5, y, fri)
    y = torch.where((y < 4.5) | (y > 7.5), y, frii)
    y = torch.where(y < 7.5, y, hybrid)
    return y


def set_requires_grad(network, bool_val):
    for p in network.parameters():
        p.requires_grad = bool_val


def parse_config(filename):
    
    config = configparser.SafeConfigParser(allow_no_value=True)
    config.read(filename)
    
    # Build a nested dictionary with tasknames at the top level
    # and parameter values one level down.
    taskvals = dict()
    for section in config.sections():
        
        if section not in taskvals:
            taskvals[section] = dict()
        
        for option in config.options(section):
            # Evaluate to the right type()
            try:
                taskvals[section][option] = ast.literal_eval(config.get(section, option))
            except (ValueError,SyntaxError):
                err = "Cannot format field '{0}' in config file '{1}'".format(option,filename)
                err += ", which is currently set to {0}. Ensure strings are in 'quotes'.".format(config.get(section, option))
                raise ValueError(err)

    return taskvals, config


def weights_init(*models):
    for m in models:
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

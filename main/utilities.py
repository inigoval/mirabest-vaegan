import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os


# define paths for saving
FILE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_PATH = os.path.join(FILE_PATH, 'files', 'figs')
IMAGE_PATH = os.path.join(FILE_PATH, 'files', 'images')
CHECKPOINT_PATH = os.path.join(FILE_PATH, 'files', 'checkpoints')

def set_requires_grad(network, bool_val):
    for p in network.parameters():
        p.requires_grad = bool_val


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def plot_losses(L_dict, epoch):
    fig, ax = plt.subplots(1,1)
    x_plot = L_dict['x_plot']
    half = torch.ones(x_plot.shape[0])/2

    ## plot losses for each network ##
    ax.plot(x_plot[:epoch], L_dict['L_E'][:epoch], label='encoder loss')
    ax.plot(x_plot[:epoch], L_dict['L_G'][:epoch], label='decoder loss')
    ax.plot(x_plot[:epoch], L_dict['L_D'][:epoch], label = 'discriminator loss')
    ax.set_xlabel('epoch')
    ax.legend()
    fig.savefig(FIG_PATH + '/network_losses.pdf')
    plt.close(fig)

    ## plot average predicted probabilities for generated/reconstructed images ##
    fig, ax = plt.subplots(1,1)
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
    X_gen = G(Z_plot).cpu().detach().view(-1,150,150).numpy()
    # set up a 2x2 grid of plots in figure 2
    # [ax21  ax22]
    # [ax23  ax24]
    fig, ((ax21, ax22), (ax23, ax24)) = plt.subplots(2,2)
    im1 = ax21.imshow(X_gen[0,:,:], cmap='hot')
    im2 = ax22.imshow(X_gen[1,:,:], cmap='hot')
    im3 = ax23.imshow(X_gen[2,:,:], cmap='hot')
    im4 = ax24.imshow(X_gen[3,:,:], cmap='hot')
    fig.savefig(IMAGE_PATH + '/epoch_{}.pdf'.format(epoch))
    plt.close(fig)
    
    X_sample = X[:2, :, :,:]
    X_recon = G(E(X_sample)[0]).cpu().detach().view(-1,150,150).numpy()
    # set up a 2x2 grid of plots in figure 2
    # [ax21  ax22]
    # [ax23  ax24]
    fig, ((ax21, ax22), (ax23, ax24)) = plt.subplots(2,2)
    im1 = ax21.imshow(X_recon[0,:,:], cmap='hot')
    im2 = ax22.imshow(X_recon[1,:,:], cmap='hot')
    im3 = ax23.imshow(X_sample[0,:,:].cpu().view(150,150).numpy(), cmap='hot')
    im4 = ax24.imshow(X_sample[1,:,:].cpu().view(150,150).numpy(), cmap='hot')
    fig.savefig(IMAGE_PATH + '/recon_epoch_{}.pdf'.format(epoch))
    plt.close(fig)



def add_noise(bool_val, X, noise_scale, epoch, n_epochs):
    """
    If bool_val == True, add noise to the input data, otherwise leave it unchanged
    """
    if bool_val == True:
        epsilon = torch.clamp(0.75- epoch/n_epochs, min=0, max=1)
        X_noisey = X + torch.randn_like(X)*epsilon*noise_scale
        X_noisey = torch.clamp(X_noisey, -1,1)
        return X_noisey
    else:
        return X

def p_flip_ann(epoch, n_epochs):
    p_flip = 1 - torch.FloatTensor(np.array(epoch/(n_epochs*0.5)))
    p_flip = torch.clamp(p_flip, 0,1)
    return p_flip

def labels(label_flip, smoothing, p_flip, smoothing_scale, n_X):
    ones = torch.ones(n_X, dtype = torch.float).cuda()
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

# define losses
scalar_tensor = torch.ones(1)
def KL_loss(mu, logvar):
    kl = -0.5 * torch.mean(1  + logvar - logvar.exp() - mu.pow(2)).view(1)
    #print('kl loss: ', kl)
    assert kl.size() == scalar_tensor.size()
    return kl

# sample z using random sample from normal distribution (epsilon)
def z_sample(mu, logvar):
    std = logvar.exp().pow(0.5)
    epsilon = torch.randn_like(std) # returns tensor same size as std but mean 0 var 1
    z_tilde = torch.add(mu, torch.mul(epsilon, std))
    #print(z_tilde.requires_grad)
    assert z_tilde.size() == mu.size()
    return z_tilde
import torch
import matplotlib.pyplot as plt
import torchvision
import time
import torch.nn as nn
import numpy as np
import os
import glob
import yaml

from networks import enc, dec, disc, I
from utilities import Annealed_Noise, Labels, Plot_Images, Set_Model, load_config, KL_loss, z_sample, plot_losses, plot_images
from dataloading import load_data
from evaluation import FID, Eval

# define paths for saving
FILE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EVAL_PATH = os.path.join(FILE_PATH, 'files', 'eval')
DATA_PATH = os.path.join(FILE_PATH, 'data')
CHECKPOINT_PATH = os.path.join(FILE_PATH, 'files', 'checkpoints')
FIG_PATH = os.path.join(FILE_PATH, 'files', 'figs')
IMAGE_PATH = os.path.join(FILE_PATH, 'files', 'images')
CONFIG_PATH = os.path.join(FILE_PATH, 'configs')

EMBEDDING_PATH = os.path.join(EVAL_PATH, 'embeddings')
FAKE_PATH = os.path.join(IMAGE_PATH, 'fake')
RECON_PATH = os.path.join(IMAGE_PATH, 'reconstructed')

EMBEDDING_PATH_REAL = os.path.join(EMBEDDING_PATH, 'real')
EMBEDDING_PATH_FAKE = os.path.join(EMBEDDING_PATH, 'fake')

path_list = [FILE_PATH, EVAL_PATH, DATA_PATH, CHECKPOINT_PATH, FIG_PATH,
             IMAGE_PATH, EMBEDDING_PATH, FAKE_PATH, RECON_PATH, EMBEDDING_PATH_FAKE,
             EMBEDDING_PATH_REAL]


# Create any directories that don't exist
for path in path_list:
    if not os.path.exists(path):
        os.makedirs(path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda = torch.device('cuda')

config = load_config()

## Load parameters from config ##
batch_size = config['data']['batch_size']
fraction = config['data']['fraction']

n_z = config['model']['n_z']

n_epochs = config['training']['n_epochs']
n_cycles = config['training']['n_cycles']
gamma = config['training']['gamma']  
noise_scale = config['training']['noise_scale']
p_flip = config['training']['p_flip']
label = config['training']['label']
lr = config['training']['lr']


## initialise networks, losses and optimizers ##
E, G, D = enc().cuda(), dec().cuda(), disc().cuda()
E_opt = torch.optim.Adam(E.parameters(), lr=lr)
G_opt = torch.optim.Adam(G.parameters(), lr=lr)
D_opt = torch.optim.Adam(D.parameters(), lr=lr)
MSE_loss = nn.MSELoss(reduction='mean')
BCE_loss = nn.BCELoss(reduction='mean')


## load and normalise data ##
trainLoader, testLoader, n_test = load_data(batch_size, label=label, fraction=fraction)

# assign dictionary to hold plotting values
L_dict = {'x_plot': np.arange(n_epochs), 'y_gen': torch.zeros(n_epochs), 'y_recon': torch.zeros(n_epochs)}


# Load pretrained classifier model
I = torch.load(EVAL_PATH + '/I.pt').cpu().eval()

# Load full datasets for evaluation
X_full, y_full = load_data(batch_size, label=label, tensor=True, fraction=fraction)

# Initialise some metrics
samples = 0
best_fid = 100 

# Initialise classes
noise = Annealed_Noise(noise_scale)
img_grid = Plot_Images(X_full, n_z, grid_length=6)
Epoch_Plot = Plot_Images(X_full, n_z, grid_length=2)
fid = FID(I, X_full)
labels = Labels(p_flip)
eval = Eval(n_epochs)


for epoch in range(n_epochs):

    # Set models to train mode #
    Set_Model.train(E, G, D)

    # Update annealed noise amplitude #
    noise.update_epsilon(epoch, n_epochs)
    eval.epsilon[epoch] = noise.epsilon

    # Initialise metrics #
    y_recon, y_gen = 0, 0

    for j in np.arange(n_cycles):
        for i, data in enumerate(trainLoader, 0):
            X, _ = data
            n_X = X.shape[0]
            samples += n_X
            X = X.cuda()

            #if i == 2:
            #   break

            # Check X is normalised properly
            if torch.max(X.pow(2)) > 1:
                print('Pixel above magnitude 1, data may not be normalised correctly')

            zeros = labels.zeros(n_X)
            ones = labels.ones(n_X)
           
            # Encode data point and reconstruct from latent space
            mu, logvar = E(X.detach())
            X_tilde = G(z_sample(mu.detach(), logvar.detach()))

            # Generate random data point
            Z = torch.randn_like(mu)
            X_p = G(Z)

            ############################################
            ########### update discriminator ###########
            ############################################
            D_opt.zero_grad()

            # Calculate gradients
            L_D_real = D.backprop(noise(X), ones, BCE_loss, retain_graph=True)
            L_D_fake = D.backprop(noise(X_p).detach(), zeros, BCE_loss, retain_graph=True)
            L_D_recon = D.backprop(noise(X_tilde).detach(), zeros, BCE_loss)

            ## Combine gradients and step optimizer ##
            L_D = (L_D_real + L_D_fake + L_D_recon)/3
            D_opt.step()

            ############################################
            ############# update decoder ###############
            ############################################
            G_opt.zero_grad()

            ## Reconstructed batch ##
            X_tilde = G(z_sample(mu.detach(), logvar.detach()))
            y_X_tilde, D_l_recon = D(noise(X_tilde))
            y_X_tilde = y_X_tilde.view(-1)
            L_G_recon = G.backprop(y_X_tilde, ones, BCE_loss)
            # Average and save output probability
            y_recon += y_X_tilde.mean().item()

            ## Fake batch ##
            y_X_p = D(noise(X_p))[0].view(-1)
            L_G_fake = G.backprop(y_X_p, ones, BCE_loss)
            # Average and save output probability
            y_gen += y_X_p.mean().item()  

            ## VAE loss ##
            _, D_l_real = D(X)
            L_G_llike = MSE_loss(D_l_real, D_l_recon)*gamma 
            L_G_llike.backward()            

            ## Sum gradients and step optimizer ##
            L_G = (L_G_fake + L_G_recon)/2 + L_G_llike
            G_opt.step()

            ############################################
            ############# update encoder ###############
            ############################################
            E_opt.zero_grad()

            ## KL loss ##
            mu, logvar = E(X)
            L_E_KL = KL_loss(mu, logvar)
            L_E_KL.backward(retain_graph=True)

            ## llike loss ##
            # Forward passes to generate feature maps
            X_tilde = G(z_sample(mu, logvar))
            _, D_l_recon = D(X_tilde)
            _, D_l_real = D(X)
            L_E_llike = MSE_loss(D_l_real, D_l_recon)
            L_E_llike.backward()

            ## Sum losses and step optimizer ##
            L_E = L_E_llike + L_E_KL
            E_opt.step()

            iterations = i

    ## Insert cumulative losses into dictionary ##
    L_dict['y_gen'][epoch] = y_gen/(iterations*n_cycles)
    L_dict['y_recon'][epoch] = y_recon/(iterations*n_cycles)

    #if (epoch+1) % 10 == 0:
        #torch.save(D, CHECKPOINT_PATH + f'/D_fr{label+1}_{epoch}.pt')
        #torch.save(L_dict, EVAL_PATH + '/L_dict.pt')

    with torch.no_grad():
        # Set models to evaluation mode
        Set_Model.eval(E, G, D)

        ## Plot image grid at regular intervals ##
        if (epoch+1) % 10 == 0:
            img_grid.plot_generate(E, G, filename=f'grid_X_recon_{epoch+1}.pdf', recon=True)
            img_grid.plot_generate(E, G, filename=f'grid_X_fake_{epoch+1}.pdf', recon=False)

        plot_images(X, E, G, n_z, epoch)

        # Generate a set of fake images and calculate FID #
        X_fake = fid.generate(G, n_z, n_samples=1000).cpu()
        score = fid.calculate_fid(X_fake)

        # Update metrics #
        eval.samples[epoch] = samples
        eval.fid[epoch] = score
        eval.calc_overfitscore(D, testLoader, n_test, noise)
        eval.calc_ratio(X_fake, I)

        # Plot metrics #
        eval.plot_fid( eps=True, ylim=1000)
        eval.plot_fid(ylim=400)
        eval.plot_fid(ylim=100)
        eval.plot_overfitscore()


        print(f'epoch {epoch+1}/{n_epochs}  |  samples:{samples}  |  FID {score:.3f}  |  best: {best_fid:.1f}')

        # Save generator/encoder weights if FID score is high
        if score < best_fid:
            print('Model saved')
            best_fid = int(score)
            img_grid.plot_generate(E, G, filename=f'grid_X_recon_{epoch+1}.pdf', recon=True)
            img_grid.plot_generate(E, G, filename=f'grid_X_fake_{epoch+1}.pdf', recon=False)

            torch.save({'epoch': epoch,
                       'G': G.state_dict(),
                       'E': E.state_dict(),
                       'FID': score,},
                       CHECKPOINT_PATH + f'/model_dict_fr{label+1}_e{epoch+1}_fid{int(score)}.pt')



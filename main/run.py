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
from utilities import Annealed_Noise, Plot_Images, Labels 
from utilities import z_sample, plot_images, KL_loss, load_config
from evaluation import FID, Eval
from dataloading import load_data

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
skip = config['training']['skip']

## initialise parameters ##
# smoothing parameters
smoothing = False
smoothing_scale = 0.12
# discriminator noise parameters
noise = True
noise_scale = 0.35
# label flipping
label_flip = True
# image grid plotting
n_images = 6
# 0 = fri 1 = frii

## initialise networks, losses and optimizers ##
E, G, D = enc().cuda(), dec().cuda(), disc().cuda()
E_opt = torch.optim.Adam(E.parameters(), lr=lr)
G_opt = torch.optim.Adam(G.parameters(), lr=lr)
D_opt = torch.optim.Adam(D.parameters(), lr=lr)
MSE_loss = nn.MSELoss(reduction='mean')
BCE_loss = nn.BCELoss(reduction='mean')


## load and normalise data ##
trainLoader, testLoader, n_test = load_data(batch_size, label=label, fraction=fraction)

# Assign dictionary to hold plotting values
L_dict = {'x_plot': np.arange(n_epochs), 'L_E': torch.zeros(n_epochs), 'L_G': torch.zeros(n_epochs),
          'L_D': torch.zeros(n_epochs), 'y_gen': torch.zeros(n_epochs), 'y_recon': torch.zeros(n_epochs)}


# Load pretrained classifier model
I = torch.load(EVAL_PATH + '/I.pt').cpu().eval()

samples = 0
best_fid = 100

# Initialise classes
noise = Annealed_Noise(noise_scale, n_epochs)
im_grid = Plot_Images(X_full, n_z)
fid = FID(I, X_full)
eval = Eval(n_epochs)
labels = Labels(p_flip)

for epoch in range(n_epochs):
    
    # Update annealed noise amplitude #
    noise.update_epsilon(epoch)
    eval.epsilon[epoch] = noise.epsilon

    D.train()
    L_E_cum, L_G_cum, L_D_cum = 0, 0, 0
    y_recon, y_gen = 0, 0

    for j in np.arange(n_cycles):
        for i, data in enumerate(trainLoader, 0):
            X, _ = data
            n_X = X.shape[0]
            samples += n_X
            X = X.cuda()
            
            # Skip training (use to quickly test code) #
            if skip: 
                if i == 3:
                    break

            # Check X is normalised properly
            if torch.max(X.pow(2)) > 1:
                print('Pixel above magnitude 1, data may not be normalised correctly')

            n_X = X.shape[0]

            zeros = labels.zeros(n_X)
            ones = labels.ones(n_X)
            
            G = G.cuda()

            ############################################
            ########### update discriminator ###########
            ############################################
            D_opt.zero_grad()

            ## train with all real batch ##
            y_X = D(noise(X))[0]
            y_X = y_X.view(-1)
            L_D_X = BCE_loss(y_X, ones)
            L_D_X.backward()

            ## train with reconstructed batch ##
            # decode data point and reconstruct from latent space
            mu, logvar = E(X.detach())
            X_tilde = G(z_sample(mu.detach(), logvar.detach()))
            # pass through discriminator -> backprop loss
            y_X_tilde = D(noise(X_tilde).detach())[0].view(-1)
            L_D_X_tilde = BCE_loss(y_X_tilde, zeros)
            L_D_X_tilde.backward()

            ## train with random batch ##
            # sample z from p(z) = N(0,1) -> generate X
            Z = torch.randn_like(mu)
            X_p = G(Z)
            # pass through Discriminator -> backprop loss
            y_X_p = D(noise(X_p).detach())[0].view(-1)
            L_D_X_p = BCE_loss(y_X_p, zeros)
            L_D_X_p.backward()

            # sum gradients
            L_D = (L_D_X + L_D_X_p + L_D_X_tilde)/3

            # step optimizer
            # if L_D > L_G:
            D_opt.step()

            ############################################
            ############# update decoder ###############
            ############################################
            G_opt.zero_grad()

            ## train with reconstructed batch ##
            # decode data point and reconstruct from latent space
            X_tilde = G(z_sample(mu.detach(), logvar.detach()))
            # pass through Driminator -> backprop loss
            y_X_tilde, D_l_X_tilde = D(noise(X_tilde))
            y_X_tilde = y_X_tilde.view(-1)
            L_G_recon = G.backprop(y_X_tilde, ones, BCE_loss)
            # Average and save output probability
            y_recon += y_X_tilde.mean().item()

            ## train with random batch ##
            # sample z from p(z) = N(0,1) -> generate X
            X_p = G(Z)
            # pass through Driminator -> backprop loss
            y_X_p = D(noise(X_p))[0].view(-1)
            L_G_X_p = BCE_loss(y_X_p, ones)
            L_G_X_p.backward(retain_graph=True)
            y_gen += y_X_p.mean().item()  # average and save output probability

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

    with torch.no_grad():
        ## Plot image grid ##
        if (epoch+1) % 10 == 0:
            im_grid.plot_generate(E, G, filename=f'grid_X_recon_{epoch+1}.pdf', recon=True)
            im_grid.plot_generate(E, G, filename=f'grid_X_fake_{epoch+1}.pdf', recon=False)

        plot_images(X, E, G, n_z, epoch)

        # generate a set of fake images
        X_fake = FID.generate(G, n_z, n_samples=1000).cpu()

        ## Calculate and store metrics ##
        score = fid(X_fake)
        eval.samples[epoch] = samples
        eval.fid[epoch] = score
        eval.calc_overfitscore(D, testLoader, n_test, noise)
        eval.calc_ratio(X_fake, I)

        ## Plot Metrics ##
        eval.plot_fid(eps=True, ylim=1000)
        eval.plot_fid(eps=False, ylim=400)
        eval.plot_fid(eps=False, ylim=100)
        eval.plot_overfitscore()


        # Save generator/encoder weights if FID score is high
        if score < best_fid:
            print('Model saved')
            best_fid = int(score)
            im_grid.plot_generate(E, G, filename=f'grid_X_recon_{epoch+1}.pdf', recon=True)
            im_grid.plot_generate(E, G, filename=f'grid_X_fake_{epoch+1}.pdf', recon=False)
            torch.save({'epoch': epoch,
                        'G': G.state_dict(),
                        'E': E.state_dict(),
                        'FID': score,
                        }, CHECKPOINT_PATH + f'/model_dict_fr{label+1}_e{epoch+1}_fid{int(score)}.pt')

        print(f'epoch {epoch+1}/{n_epochs}  |  samples:{samples}  |  FID {score:.3f}  |  best: {best_fid:.1f}')

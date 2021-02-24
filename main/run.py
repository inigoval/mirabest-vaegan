import torch
import matplotlib.pyplot as plt
import torchvision
import time
import torch.nn as nn
import numpy as np
import os

from networks import enc, dec, disc, I
from utilities import add_noise, p_flip_ann, labels, KL_loss, z_sample, plot_losses, plot_images, plot_grid, parse_config, set_train
from dataloading import load_data
from evaluation import compute_mu_sig, fid, test_prob, ratio, plot_eval_dict, generate

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


## initialise networks, losses and optimizers ##
E, G, D = enc().cuda(), dec().cuda(), disc().cuda()
E_opt = torch.optim.Adam(E.parameters(), lr=2e-4)
G_opt = torch.optim.Adam(G.parameters(), lr=2e-4)
D_opt = torch.optim.Adam(D.parameters(), lr=2e-4)
MSE_loss = nn.MSELoss(reduction='mean')
BCE_loss = nn.BCELoss(reduction='mean')

## initialise parameters ##
batch_size = 32
n_z = 32
n_epochs = 800
n_cycles = 5
gamma = 1  # weighting for style (L_llike) in generator loss function
# smoothing parameters
smoothing = False
smoothing_scale = 0.12
# discriminator noise parameters
noise_scale = 0.35
# label flipping
label_flip = True
p_flip = 0.05
# image grid plotting
n_images = 6
# 0 = fri 1 = frii
label = 0

## load and normalise data ##
trainLoader, testLoader, n_test = load_data(batch_size, label=label, reduce=True)

# assign dictionary to hold plotting values
L_dict = {'x_plot': np.arange(n_epochs), 'L_E': torch.zeros(n_epochs), 'L_G': torch.zeros(n_epochs),
          'L_D': torch.zeros(n_epochs), 'y_gen': torch.zeros(n_epochs), 'y_recon': torch.zeros(n_epochs)}

eval_dict = {'x_plot': np.arange(n_epochs), 'n_samples': np.zeros(n_epochs),  'fid': np.zeros(n_epochs),
             'likeness': np.zeros(n_epochs), 'D_X_test': np.zeros(n_epochs), 'fri%': np.zeros(n_epochs),
             'epsilon': np.zeros(n_epochs)}


# Load pretrained classifier model
I = torch.load(EVAL_PATH + '/I.pt').cpu().eval()

# Load full datasets for evaluation
X_full, y_full = load_data(batch_size, label=label, reduce=True, array=True)


# Initialise random Z 
Z_plot = Z_plot = torch.randn(n_images**2, n_z).cuda().view(n_images**2, n_z, 1, 1)
# Compute mean and covariance matrix for real data
mu_fid, sigma_fid = compute_mu_sig(I, X_full)

# Initialise some metrics
samples = 0
best_fid = 100 


for epoch in range(n_epochs):

    # Set models to train mode #
    set_train(E, G, D)

    # Initialise annealed noise 
    add_noise = add_noise(noise_scale, epoch, n_epochs)

    # Initialise metrics
    L_E_cum, L_G_cum, L_D_cum = 0, 0, 0
    y_recon, y_gen = 0, 0
    eval_dict['epsilon'][epoch] = add_noise.epsilon

    for j in np.arange(n_cycles):
        for i, data in enumerate(trainLoader, 0):
            X, _ = data
            n_X = X.shape[0]
            samples += n_X
            X = X.cuda()

            # if i == 2:
            #    break

            # check X is normalised properly
            if torch.max(X.pow(2)) > 1:
                print('overflow')

            ones, zeros = labels(label_flip, smoothing, p_flip, smoothing_scale, n_X)


           
            # Add noise to data (ready for discriminator input)
            X_noisey = add_noise(X)

            # Encode data point and reconstruct from latent space
            mu, logvar = E(X.detach())
            X_tilde = G(z_sample(mu.detach(), logvar.detach()))
            X_tilde = add_noise(X_tilde)

            # Generate random data point
            Z = torch.randn_like(mu)
            X_p = G(Z)
            X_p = add_noise(X_p)

            ############################################
            ########### update discriminator ###########
            ############################################
            D_opt.zero_grad()

            # Calculate gradients
            L_D_real = D.backprop(X_noisey, ones, BCE_loss)
            L_D_fake = D.backprop(X_p, zeros, BCE_loss)
            L_D_recon = D.backprop(X_tilde, zeros, BCE_loss)

            ## Combine gradients and step optimizer ##
            L_D = (L_D_real + L_D_fake + L_D_recon)/3
            D_opt.step()

            ############################################
            ############# update decoder ###############
            ############################################
            G_opt.zero_grad()

            ## Reconstructed batch ##
            y_X_tilde, D_l_recon = D(X_tilde)
            y_X_tilde = y_X_tilde.view(-1)
            L_G_recon = G.backprop(y_recon, ones, BCE_loss)
            # Average and save output probability
            y_recon += y_X_tilde.mean().item()

            ## Fake batch ##
            y_X_p = D(X_p)[0].view(-1)
            L_G_fake = G.backprop(y_X_p, ones)
            # Average and save output probability
            y_gen += y_X_p.mean().item()  

            ## VAE loss ##
            D_l_real = D(X_noisey)[1]
            L_G_llike = MSE_loss(D_l_X, D_l_X_tilde)*gamma 
            L_G_llike.backward()            

            ## Sum gradients and step optimizer##
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
            _, D_l_X_tilde = D(X_tilde)
            L_E_llike = MSE_loss(D_l_real, D_l_X_tilde)
            L_E_llike.backward()

            ## Sum losses and step optimizer ##
            L_E = L_E_llike + L_E_KL
            E_opt.step()

            iterations = i

    ## insert cumulative losses into dictionary ##
    L_dict['y_gen'][epoch] = y_gen/(iterations*n_cycles)
    L_dict['y_recon'][epoch] = y_recon/(iterations*n_cycles)

    #if (epoch+1) % 10 == 0:
        #torch.save(D, CHECKPOINT_PATH + f'/D_fr{label+1}_{epoch}.pt')
        #torch.save(L_dict, EVAL_PATH + '/L_dict.pt')

    with torch.no_grad():

        ## Plot image grid at regular intervals ##
        if (epoch+1) % 10 == 0:
            plot_grid(n_z, E, G, Z_plot, epoch, n_images=6)

        plot_images(X, E, G, n_z, epoch)

        # Generate a set of fake images for evaluation
        X_fake = generate(G, n_z, n_samples=1000).cpu()

        FID = fid(I, mu_fid, sigma_fid, X_fake)
        eval_dict['n_samples'][epoch] = samples
        eval_dict['fid'][epoch] = FID
        eval_dict['D_X_test'][epoch] = test_prob(D.eval(), testLoader, n_test, add_noise)
        eval_dict['fri%'][epoch] = ratio(X_fake, I)

        plot_eval_dict(eval_dict, epoch)

        print(f'epoch {epoch+1}/{n_epochs}  |  samples:{samples}  |  FID {FID:.3f}')

        # Save generator/encoder weights if FID score is high
        if FID < best_fid:
            print('Model saved')
            best_fid = int(FID)
            plot_grid(n_z, E, G, Z_plot, epoch, n_images=6)
            torch.save({'epoch': epoch,
                        'G': G.state_dict(),
                        'E': E.state_dict(),
                        'FID': FID,
                        }, CHECKPOINT_PATH + f'/model_dict_fr{label+1}_e{epoch+1}_fid{int(FID)}.pt')



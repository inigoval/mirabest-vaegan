import torch 
import matplotlib.pyplot as plt
import torchvision
import time
import torch.nn as nn
import numpy as np
import os

from vaegan import enc, dec, disc
from utilities import labels, z_sample, add_noise, plot_losses, p_flip_ann
from utilities import plot_images, KL_loss, sparsity_loss, plot_grid
from dataloading import load_data

# define paths for saving
FILE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(FILE_PATH, 'data')
CHECKPOINT_PATH = os.path.join(FILE_PATH, 'files', 'checkpoints')

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
batch_size_test = 1000
n_z = 100
n_epochs = 1000
gamma = 1  # weighting for style (L_llike) in generator loss function
# smoothing parameters
smoothing = False
smoothing_scale = 0.12
# discriminator noise parameters
noise = True
noise_scale = 0.5
# label flipping
label_flip = False
#p_flip = 0.05
# image grid plotting
n_images = 6

## load and normalise MNIST data ## 
trainLoader, testLoader = load_data(batch_size)

# assign dictionary to hold plotting values
L_dict = {'x_plot': np.arange(n_epochs), 'L_E': torch.zeros(n_epochs), 'L_G': torch.zeros(n_epochs), 
		  'L_D': torch.zeros(n_epochs), 'y_gen': torch.zeros(n_epochs), 'y_recon': torch.zeros(n_epochs)}

# initialise noise for grid images so that latent vector is same every time
Z_plot = Z_plot = torch.randn(n_images**2, n_z).cuda().view(n_images**2, n_z, 1, 1)

for epoch in range(n_epochs):
	L_E_cum, L_G_cum, L_D_cum  = 0, 0, 0
	y_recon, y_gen = 0, 0
	p_flip = p_flip_ann(epoch, n_epochs)
	for i, data in enumerate(trainLoader , 0):
		X, _ = data
		X = X.cuda()
		
		# check X is normalised properly 
		if torch.max(X.pow(2)) > 1:
			print('overflow')

		n_X = X.shape[0]
		ones, zeros = labels(label_flip, smoothing, p_flip, smoothing_scale, n_X)
				
		############################################
		########### update discriminator ###########
		############################################
		D_opt.zero_grad()

		## train with all real batch ##
		y_X = D(add_noise(noise, X, noise_scale, epoch, n_epochs))[0]
		y_X = y_X.view(-1)
		L_D_X = BCE_loss(y_X, ones)
		L_D_X.backward()

		## train with reconstructed batch ##
		# decode data point and reconstruct from latent space
		mu, logvar = E(X.detach())
		X_tilde = G(z_sample(mu.detach(), logvar.detach()))
		# pass through discriminator -> backprop loss
		y_X_tilde = D(add_noise(noise, X_tilde, noise_scale, epoch, n_epochs).detach())[0].view(-1)
		L_D_X_tilde = BCE_loss(y_X_tilde, zeros)
		L_D_X_tilde.backward()

		## train with random batch ##
		# sample z from p(z) = N(0,1) -> generate X
		Z = torch.randn_like(mu)
		X_p = G(Z)
		# pass through Discriminator -> backprop loss
		y_X_p = D(add_noise(noise, X_p, noise_scale, epoch, n_epochs).detach())[0].view(-1)
		L_D_X_p = BCE_loss(y_X_p, zeros)
		L_D_X_p.backward()

		## sum gradients
		L_D = (L_D_X + L_D_X_p + L_D_X_tilde)/3

		## step optimizer
		#if L_D > L_G:
		D_opt.step()

		############################################
		############# update decoder ###############
		############################################

		G_opt.zero_grad()
		
		## train with reconstructed batch ##
		# decode data point and reconstruct from latent space
		# ------>>>>  mu, logvar = E(X.detach()) <<< ---------
		X_tilde = G(z_sample(mu.detach(), logvar.detach()))
		# pass through Driminator -> backprop loss
		y_X_tilde, D_l_X_tilde = D(add_noise(noise, X_tilde, noise_scale, epoch, n_epochs))
		y_X_tilde = y_X_tilde.view(-1)
		L_G_X_tilde = BCE_loss(y_X_tilde, ones)
		L_G_X_tilde.backward(retain_graph=True)
		y_recon += y_X_tilde.mean().item() # average and save output probability

		## train with random batch ##
		# sample z from p(z) = N(0,1) -> generate X
		X_p = G(Z)
		# pass through Driminator -> backprop loss
		y_X_p = D(add_noise(noise, X_p, noise_scale, epoch, n_epochs))[0].view(-1)
		L_G_X_p = BCE_loss(y_X_p, ones)
		L_G_X_p.backward(retain_graph=True)
		y_gen += y_X_p.mean().item() # average and save output probability

		## VAE loss ##
		D_l_X = D(X)[1]
		L_G_llike = MSE_loss(D_l_X, D_l_X_tilde)*gamma # maybe detach D output here
		L_G_llike.backward()

		## sum gradients ##
		L_G = (L_G_X_p + L_G_X_tilde)/2 + L_G_llike

		## step optimizer
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
		# forward passes to generate feature maps
		X_tilde = G(z_sample(mu, logvar))
		_, D_l_X_tilde = D(X_tilde)
		_, D_l_X = D(X)
		L_E_llike = MSE_loss(D_l_X, D_l_X_tilde)
		L_E_llike.backward()
		# sum losses
		L_E = L_E_llike + L_E_KL

		# step optimizer
		E_opt.step()

		# update cumulative losses
		L_E_cum += L_E.item()
		L_G_cum += L_G.item()
		L_D_cum += L_D.item()

		iterations = i

	## insert cumulative losses into dictionary ##
	L_dict['L_E'][epoch] = L_E_cum/iterations
	L_dict['L_G'][epoch] = L_G_cum/iterations
	L_dict['L_D'][epoch] = L_D_cum/iterations
	L_dict['y_gen'][epoch] = y_gen/iterations
	L_dict['y_recon'][epoch] = y_recon/iterations

	if epoch % 10 == 0:
		torch.save(E, CHECKPOINT_PATH + '/E_{:f}.pt'.format(epoch))
		torch.save(G, CHECKPOINT_PATH + '/G_{:f}.pt'.format(epoch))
		torch.save(D, CHECKPOINT_PATH + '/D_{:f}.pt'.format(epoch))
		torch.save(L_dict, CHECKPOINT_PATH + '/L_dict.pt')

	## plot and save losses/images ##
	if epoch % 1 ==0:
		plot_losses(L_dict, epoch)
		plot_images(X, E, G, n_z, epoch)

	## plot image grid ##
	if epoch % 5 == 0:
		plot_grid(n_z, E, G, Z_plot, epoch, n_images=6)

	print('epoch {}/{}  |  L_E {:.4f}  |  L_G {:.4f}  |  L_D {:.4f}  |  y_gen {:.3f}  |  y_recon {:.3f}'.format(epoch, n_epochs, L_E_cum/iterations, L_G_cum/iterations, L_D_cum/iterations, y_gen/iterations, y_recon/iterations))
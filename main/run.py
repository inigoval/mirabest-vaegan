import torch
import torchvision
import torch.nn as nn
import numpy as np

from pathlib import Path

from networks import enc, dec, disc, I
from utilities import Annealed_Noise, Plot_Images, Labels, Set_Model
from utilities import z_sample, KL_loss, load_config
from evaluation import FID, Eval
from dataloading import Data_Agent
from paths import Path_Handler
from datasets import MiraBest_full, MB_nohybrids, MBFRConfident

paths = Path_Handler()
path_dict = paths._dict()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda = torch.device('cuda')

config = load_config()

# Load parameters from config #
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

# Initialise networks, losses and optimizers #
E, G, D = enc().cuda(), dec().cuda(), disc().cuda()
E_opt = torch.optim.Adam(E.parameters(), lr=lr)
G_opt = torch.optim.Adam(G.parameters(), lr=lr)
D_opt = torch.optim.Adam(D.parameters(), lr=lr)
MSE_loss = nn.MSELoss(reduction='mean')
BCE_loss = nn.BCELoss(reduction='mean')

# Load data #
data_agent = Data_Agent(MB_nohybrids)
data_agent.set_labels(label)
data_agent.subset(fraction)
data_agent.fid_dset()

trainLoader, testLoader = data_agent.load()

# Assign dictionary to hold plotting values
L_dict = {
    'x_plot': np.arange(n_epochs),
    'L_E': torch.zeros(n_epochs),
    'L_G': torch.zeros(n_epochs),
    'L_D': torch.zeros(n_epochs),
    'y_gen': torch.zeros(n_epochs),
    'y_recon': torch.zeros(n_epochs)
}

# Load inception model
I = torch.load(path_dict['eval'] / 'I.pt').cpu().eval()

samples = 0
best_fid = 50
best_epoch = None

# Initialise annealed noise #
noise = Annealed_Noise(noise_scale, n_epochs)

# Initialise image grids #
im_grid = Plot_Images(data_agent.X_fid, n_z, path_dict)
im_stamp = Plot_Images(data_agent.X_fid, n_z, path_dict, grid_length=2)

aug_list = [] 
aug_list.append(Plot_Images(data_agent.X_fid, n_z, path_dict, grid_length=4, alpha=0.5))
aug_list.append(Plot_Images(data_agent.X_fid, n_z, path_dict, grid_length=4, alpha=1))
aug_list.append(Plot_Images(data_agent.X_fid, n_z, path_dict, grid_length=4, alpha=2))

# Initialise evaluation classes #
fid = FID(I, data_agent.X_fid)
eval = Eval(n_epochs)

# Initialise label generator #
labels = Labels(p_flip)

for epoch in range(n_epochs):

    # Update annealed noise amplitude #
    noise.update_epsilon(epoch)
    eval.epsilon[epoch] = noise.epsilon

    # D.train()
    Set_Model.train(G, D)
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
                if i == 2:
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
            L_D = (L_D_X + L_D_X_p + L_D_X_tilde) / 3

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
            L_G_X_tilde = BCE_loss(y_X_tilde, ones)
            L_G_X_tilde.backward(retain_graph=True)
            y_recon += y_X_tilde.mean().item(
            )  # average and save output probability

            ## train with random batch ##
            # sample z from p(z) = N(0,1) -> generate X
            X_p = G(Z)
            # pass through Driminator -> backprop loss
            y_X_p = D(noise(X_p))[0].view(-1)
            L_G_X_p = BCE_loss(y_X_p, ones)
            L_G_X_p.backward(retain_graph=True)
            y_gen += y_X_p.mean().item()  # average and save output probability

            ## VAE loss ##
            D_l_X = D(X)[1]
            L_G_llike = MSE_loss(
                D_l_X, D_l_X_tilde) * gamma  # maybe detach D output here
            L_G_llike.backward()

            ## sum gradients ##
            L_G = (L_G_X_p + L_G_X_tilde) / 2 + L_G_llike

            # step optimizer
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
            _, D_l_X = D(X)
            L_E_llike = MSE_loss(D_l_X, D_l_X_tilde)
            L_E_llike.backward()

            ## Sum losses and step optimizer ##
            L_E = L_E_llike + L_E_KL
            E_opt.step()

            iterations = i

    ## insert cumulative losses into dictionary ##
    L_dict['L_E'][epoch] = L_E_cum / (iterations * n_cycles)
    L_dict['L_G'][epoch] = L_G_cum / (iterations * n_cycles)
    L_dict['L_D'][epoch] = L_D_cum / (iterations * n_cycles)
    L_dict['y_gen'][epoch] = y_gen / (iterations * n_cycles)
    L_dict['y_recon'][epoch] = y_recon / (iterations * n_cycles)

    with torch.no_grad():
        Set_Model.eval(G, D)
        ## Plot image grid ##
        if (epoch + 1) % 10 == 0:
            im_grid.plot_generate(E, G, filename=f'grid_X_recon_{epoch+1}.pdf', recon=True)
            im_grid.plot_generate(E, G, filename=f'grid_X_fake_{epoch+1}.pdf', recon=False)

            for aug in aug_list:
                aug.GAug(E, G)
                aug.plot(path_dict['gaug'] / f'epoch{epoch+1}_alpha{aug.alpha}.png')

        im_stamp.plot_generate(E, G, filename=f'X_fake_{epoch+1}.pdf', recon=False)
        im_stamp.plot_generate(E, G, filename=f'X_recon_{epoch+1}.pdf', recon=True)

        # generate a set of fake images
        X_fake = FID.generate(G, n_z, n_samples=1000).cpu()

        ## Calculate and store metrics ##
        score = fid(X_fake)
        eval.samples[epoch] = samples
        eval.fid[epoch] = score
        eval.calc_overfitscore(D, testLoader, data_agent.n_test, noise)
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
            best_epoch = epoch + 1
            im_grid.plot_generate(E, G, filename=f'grid_X_recon_{epoch+1}.pdf', recon=True)
            im_grid.plot_generate(E, G, filename=f'grid_X_fake_{epoch+1}.pdf', recon=False)
            for aug in aug_list:
                aug.GAug(E, G)
                aug.plot(path_dict['gaug'] / f'epoch{epoch+1}_alpha{aug.alpha}.png')

            torch.save(
                {
                    'epoch': epoch,
                    'G': G.state_dict(),
                    'E': E.state_dict(),
                    'FID': score,
                },
                path_dict['checkpoints'] / f'model_dict_fr{label+1}_e{epoch+1}_fid{int(score)}.pt')

        print(
            f'epoch {epoch+1}/{n_epochs}  |  samples:{samples}  |  FID {score:.3f}  |  best: {best_fid:.1f} (epoch {best_epoch})'
        )

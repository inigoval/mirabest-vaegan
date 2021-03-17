import torch
import torchvision
import torch.nn as nn
import numpy as np

from pathlib import Path
from sacred import Experiment

from networks import enc, dec, disc, I
from utilities import Annealed_Noise, Plot_Images, Labels, Set_Model, Plot_GAug
from utilities import z_sample, KL_loss, load_config, plot_hist
from evaluation import FID, Eval
from dataloading import Data_Agent
from paths import Path_Handler
from datasets import MiraBest_full, MB_nohybrids, MBFRConfident


paths = Path_Handler()
path_dict = paths._dict()

ex = Experiment()

config = load_config()

# Load parameters from config #
batch_size = config["data"]["batch_size"]
fraction = config["data"]["fraction"]

n_z = config["model"]["n_z"]
model_type = config["model"]["architecture"]

n_epochs = config["training"]["n_epochs"]
n_cycles = config["training"]["n_cycles"]
gamma = config["training"]["gamma"]
noise_scale = config["training"]["noise_scale"]
p_flip = config["training"]["p_flip"]
label = config["training"]["label"]
lr = config["training"]["lr"]
skip = config["training"]["skip"]
T = config["training"]["T"]
seed = config["training"]["seed"]

# Initialise networks, losses and optimizers #
E, G, D = enc().cuda(), dec(model_type=model_type).cuda(), disc().cuda()
E_opt = torch.optim.Adam(E.parameters(), lr=lr)
G_opt = torch.optim.Adam(G.parameters(), lr=lr)
D_opt = torch.optim.Adam(D.parameters(), lr=lr)
MSE_loss = nn.MSELoss(reduction="mean")
BCE_loss = nn.BCELoss(reduction="mean")
Set_Model.weights_init(E, G, D)

# Load data #
data_agent = Data_Agent(MB_nohybrids, seed=seed)
data_agent.subset(fraction)
data_agent.set_labels(label)
data_agent.fid_dset()

trainLoader, testLoader = data_agent.load()

# Assign dictionary to hold plotting values
L_dict = {
    "x_plot": np.arange(n_epochs),
    "L_E": torch.zeros(n_epochs),
    "L_G": torch.zeros(n_epochs),
    "L_D": torch.zeros(n_epochs),
    "y_gen": torch.zeros(n_epochs),
    "y_recon": torch.zeros(n_epochs),
}

# Load inception model
I = torch.load(path_dict["eval"] / "I.pt").cpu().eval()

samples = 0
best_fid = 100
best_epoch = None

# Initialise annealed noise #
noise = Annealed_Noise(noise_scale, n_epochs)

# Initialise image grids #
im_grid = Plot_Images(data_agent.X_fid, n_z, path_dict)
im_stamp = Plot_Images(data_agent.X_fid, n_z, path_dict, grid_length=2)

# Initialise augmentation plotting
im_aug = Plot_GAug(data_agent.X_fid, path_dict)

# Initialise evaluation classes #
fid_fake = FID(I, data_agent.X_fid)
fid_recon = FID(I, data_agent.X_fid[:1000, ...])
eval = Eval(n_epochs)

# Initialise label generator #
labels = Labels(p_flip)

for epoch in range(n_epochs):

    # Update annealed noise amplitude #
    noise.update_epsilon(epoch)
    eval.update_epoch(epoch)
    eval.epsilon[epoch] = noise.epsilon

    # D.train()
    Set_Model.train(E, G, D)
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
                print("overflow")

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
            _, D_l_X = D(noise(X, update_noise=False))
            y_X_tilde = y_X_tilde.view(-1)
            L_G_X_tilde = BCE_loss(y_X_tilde, ones)
            L_G_X_tilde.backward(retain_graph=True)
            y_recon += y_X_tilde.mean().item()  # average and save output probability

            ## train with random batch ##
            # sample z from p(z) = N(0,1) -> generate X
            X_p = G(Z)
            # pass through Driminator -> backprop loss
            y_X_p = D(noise(X_p))[0].view(-1)
            L_G_X_p = BCE_loss(y_X_p, ones)
            L_G_X_p.backward(retain_graph=True)
            y_gen += y_X_p.mean().item()  # average and save output probability

            ## VAE loss ##
            L_G_llike = MSE_loss(D_l_X, D_l_X_tilde) * gamma
            L_G_llike.backward()

            ## Sum gradients ##
            L_G = (L_G_X_p + L_G_X_tilde) / 2 + L_G_llike

            # step optimizer
            G_opt.step()

            ############################################
            ############# update encoder ###############
            ############################################
            E_opt.zero_grad()

            ## KL loss ##
            mu, logvar = E(X)
            L_E_KL = T * KL_loss(mu, logvar)
            L_E_KL.backward(retain_graph=True)

            ## llike loss ##
            # Forward passes to generate feature maps
            X_tilde = G(z_sample(mu, logvar))
            _, D_l_X_tilde = D(noise(X_tilde))
            _, D_l_X = D(noise(X, update_noise=False))
            L_E_llike = MSE_loss(D_l_X, D_l_X_tilde)
            L_E_llike.backward()

            ## Sum losses and step optimizer ##
            L_E = L_E_llike + L_E_KL
            E_opt.step()

            iterations = i

    ## insert cumulative losses into dictionary ##
    L_dict["L_E"][epoch] = L_E_cum / (iterations * n_cycles)
    L_dict["L_G"][epoch] = L_G_cum / (iterations * n_cycles)
    L_dict["L_D"][epoch] = L_D_cum / (iterations * n_cycles)
    L_dict["y_gen"][epoch] = y_gen / (iterations * n_cycles)
    L_dict["y_recon"][epoch] = y_recon / (iterations * n_cycles)

    with torch.no_grad():
        Set_Model.eval(E, G, D)
        ## Plot image grid ##
        if (epoch + 1) % 10 == 0:
            im_grid.plot_generate(
                E, G, filename=f"grid_X_recon_{epoch+1}.pdf", recon=True
            )
            im_grid.plot_generate(
                E, G, filename=f"grid_X_fake_{epoch+1}.pdf", recon=False
            )

            plot_hist(E, data_agent.X_fid[:1000, ...], epoch, eval=False)
            plot_hist(E, data_agent.X_fid[:1000, ...], epoch, eval=True)

        im_stamp.plot_generate(E, G, filename=f"X_fake_{epoch+1}.pdf", recon=False)
        im_stamp.plot_generate(E, G, filename=f"X_recon_{epoch+1}.pdf", recon=True)

        # generate a set of fake images
        X_fake = FID.generate(G, n_z, n_samples=1000).cpu()
        X_recon = FID.reconstruct(data_agent.X_fid, E, G, n_samples=1000).cpu()

        ## Calculate and store metrics ##
        score_fake = fid_fake(X_fake)
        score_recon = fid_recon(X_recon)
        eval.samples[epoch] = samples
        eval.calc_overfitscore(D, testLoader, data_agent.n_test, noise)
        eval.calc_ratio(X_fake, I)
        eval.fid["fake"][epoch] = score_fake
        eval.fid["recon"][epoch] = score_recon

        ## Plot Metrics ##
        eval.plot_fid(eps=True, ylim=1000, fid_type="fake")
        eval.plot_fid(eps=False, ylim=400, fid_type="fake")
        eval.plot_fid(eps=False, ylim=100, fid_type="fake")

        eval.plot_fid(eps=True, ylim=1000, fid_type="recon")
        eval.plot_fid(eps=False, ylim=400, fid_type="recon")
        eval.plot_fid(eps=False, ylim=100, fid_type="recon")

        eval.plot_overfitscore()

        # Save generator/encoder weights if FID score is high
        if score_recon < best_fid:
            print("Model saved")
            best_fid = int(score_recon)
            best_epoch = epoch + 1
            im_grid.plot_generate(
                E, G, filename=f"grid_X_recon_{epoch+1}.pdf", recon=True
            )
            im_grid.plot_generate(
                E, G, filename=f"grid_X_fake_{epoch+1}.pdf", recon=False
            )

            for alpha in np.linspace(0, 1, 5):
                im_aug.GAug(E, G, alpha=alpha, idx=0)
                im_aug.plot(epoch)

                im_aug.GAug(E, G, alpha=alpha, idx=1)
                im_aug.plot(epoch)

                im_aug.GAug(E, G, alpha=alpha, idx=2)
                im_aug.plot(epoch)

            torch.save(
                {
                    "epoch": epoch,
                    "G": G.state_dict(),
                    "E": E.state_dict(),
                    "FID": score_recon,
                },
                path_dict["checkpoints"]
                / f"model_dict_fr{label+1}_e{epoch+1}_fid{int(score_recon)}.pt",
            )

        print(
            f"epoch {epoch+1}/{n_epochs}  |  samples:{samples}  |  FID (recon) {score_recon:.3f}  |  FID (fake) {score_fake:.3f}  |  best: {best_fid:.1f} (epoch {best_epoch})"
        )

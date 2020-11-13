import torch
import numpy as np
import matplotlib.pyplot as plt
import os

FILE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_PATH = os.path.join(FILE_PATH, 'files', 'figs')
IMAGE_PATH = os.path.join(FILE_PATH, 'files', 'images')
CHECKPOINT_PATH = os.path.join(FILE_PATH, 'files', 'checkpoints')

cuda = torch.device('cuda') 

G = torch.load(CHECKPOINT_PATH + '/dec.pt')

l_z = 100

for i in range(4):
    Z_plot = torch.normal(torch.zeros(4,l_z),torch.ones(4,l_z)).cuda().view(4,l_z,1,1)
    X_synth = G(Z_plot).cpu().detach().view(4, 28,28).numpy()
    fig2, ((ax21, ax22), (ax23, ax24)) = plt.subplots(2,2)
    im1 = ax21.imshow(X_synth[0,:,:], cmap='gray_r')
    im2 = ax22.imshow(X_synth[1,:,:], cmap='gray_r')
    im3 = ax23.imshow(X_synth[2,:,:], cmap='gray_r')
    im4 = ax24.imshow(X_synth[3,:,:], cmap='gray_r')
    plt.savefig(IMAGE_PATH + '/images_{}.pdf'.format(i))
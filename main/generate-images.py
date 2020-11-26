import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1 import ImageGrid
from dataloading import load_data, MiraBest_full

FILE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(FILE_PATH, 'data')
FIG_PATH = os.path.join(FILE_PATH, 'files', 'figs')
IMAGE_PATH = os.path.join(FILE_PATH, 'files', 'images')
CHECKPOINT_PATH = os.path.join(FILE_PATH, 'files', 'checkpoints')

cuda = torch.device('cuda') 

E = torch.load(CHECKPOINT_PATH + '/E.pt').cuda()
G = torch.load(CHECKPOINT_PATH + '/G.pt').cuda()

n_z = 100
n_images = 6
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
    X_p = G(z).cpu().detach().view(150,150)
    img_list_p.append(X_p)

## generate reconstructed image list ## 
for i in range(int(n_images**2/2)):
    X = data.__getitem__(i)[0].cuda().view(1, 1, 150, 150)
    z = E(X)[0].view(1, n_z, 1, 1)
    X_tilde = G(z).cpu().detach().view(150,150)
    img_list_tilde.append(X_tilde.cpu().detach().view(150,150))
    img_list_data.append(X.cpu().detach().view(150,150))

img_list_joined = img_list_tilde + img_list_data

fig = plt.figure(figsize=(13., 13.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(n_images, n_images),  # creates 2x2 grid of axes
                 axes_pad=0,  # pad between axes in inch.
                 )

for ax, im in zip(grid, img_list_p):
    ax.imshow(im)
plt.savefig(IMAGE_PATH + '/grid_X_p.pdf')
plt.close(fig)

fig = plt.figure(figsize=(13., 13.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(n_images, n_images),  # creates 2x2 grid of axes
                 axes_pad=0,  # pad between axes in inch.
                 )

for ax, im in zip(grid, img_list_joined):
    ax.imshow(im)
plt.savefig(IMAGE_PATH + '/grid_X_tilde.pdf')
plt.close(fig)

""" fig = plt.figure(figsize=(13., 13.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(n_images, n_images),  # creates 2x2 grid of axes
                 axes_pad=0,  # pad between axes in inch.
                 )

for ax, im in zip(grid, img_list_data):
    ax.imshow(im)
plt.savefig(IMAGE_PATH + '/grid_X.pdf')
plt.close(fig) """

"""for i in range(n_images**2):
    X = data.__getitem__(i)[0].cuda().view(1, 1, 150, 150)
    Z = E(X)[0].view(1, n_z, 1, 1)
    X_tilde = G(Z).cpu().detach().view(150,150)
    img_list_tilde.append(X_tilde.cpu().detach().view(150,150))
    img_list_data.append(X.cpu().detach().view(150,150))"""
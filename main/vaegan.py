import torchvision
import numpy as np
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import os

# latent vector length
n_z = 2

# feature map sizes
n_ef = 16
n_gf = 32
n_df = 32

# number of channels
n_channels = 1
batch_size = 32

class enc(nn.Module):
    # Conv2D(in_channels, out_channels, kernel size, stride, padding)
    # conv1 (1, 150, 150)  ->  (16, 75, 75)
    # conv2 (16, 75, 75)   ->  (32, 36, 36)
    # conv3 (32, 36, 36)   ->  (64, 18, 18)
    # conv4 (64, 16, 16)   ->  (128, 8, 8)
    # fc    (128*8*8)       ->  (n_z)
    def __init__(self):
        super(enc, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 7, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 6, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True))

        self.mu = nn.Linear(128*8*8, n_z, bias= False)
        self.logvar = nn.Linear(128*8*8, n_z, bias= False)

    def forward(self, x):
        # calculating the logvariance ensurees that the std is positive
        # since exp(logvar) = std^2 > 0
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x).view(-1, 128*8*8)
        mu = self.mu(x).view(-1, n_z, 1, 1)
        logvar = self.logvar(x).view(-1, n_z, 1, 1)
        return mu, logvar

class dec(nn.Module):
    # conv1 (100, 1, 1)   ->  (256, 8,  8)
    # conv2 (32, 7, 7)   ->  (16, 18, 18)
    # conv3 (16, 14, 14) ->  (1, 36, 36)
    def __init__(self):
        super(dec, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(n_z, n_gf*8, 8, 1, 0, bias = False),
            nn.BatchNorm2d(n_gf*8),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(n_gf*8, n_gf*4, 6, 2, 1, bias =False),
            nn.BatchNorm2d(n_gf*4),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(n_gf*4, n_gf*2, 4, 2, 1, bias =False),
            nn.BatchNorm2d(n_gf*2),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(n_gf*2, n_gf, 7, 2, 1, bias=False),
            nn.BatchNorm2d(n_gf),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(n_gf, 1, 4, 2, 1, bias=False),
            nn.Tanh())

    def forward(self, z):
        x = self.conv1(z)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x).view(-1, 1, 150,150)
        return x

class disc(nn.Module):
    def __init__(self):
        super(disc, self).__init__()
        # Conv2D(in_channels, out_channels, kernel size, stride, padding)

        # conv1 (1, 150, 150)  ->  (32, 75, 75)
        # conv2 (32, 64, 64)   ->  (64, 36, 36)
        # conv3 (64, 4, 4)    ->  (128, 18, 18)
        # conv4 (128, 4, 4)    ->  (256, 8, 8)
        # conv5 (256, 8, 8)    ->  (1, 1, 1)


        self.conv1 = nn.Sequential(
            nn.Conv2d(1, n_df, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(n_df, n_df*2, 7, 2, 1, bias=False),
            nn.BatchNorm2d(n_df*2),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv3 = nn.Sequential(
            nn.Conv2d(n_df*2, n_df*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_df*4),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv4 = nn.Sequential(
            nn.Conv2d(n_df*4, n_df*8, 6, 2, 1, bias=False),
            nn.BatchNorm2d(n_df*8),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv5 = nn.Sequential(
            nn.Conv2d(n_df*8, 1, 8, 1, 0, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        D_l = x.clone()
        x = self.conv4(x)
        y_pred = self.conv5(x).view(-1)
        return y_pred, D_l
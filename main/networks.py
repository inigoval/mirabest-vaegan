import torchvision
import numpy as np
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import os

# latent vector length
n_z = 32

# feature map sizes
n_ef = 16
n_gf = 16
n_df = 16

# number of channels
n_channels = 1
batch_size = 32


class enc(nn.Module):
    # Conv2D(in_channels, out_channels, kernel size, stride, padding)
    # conv1 (1, 150, 150)  ->  (16, 76, 76)
    # conv2 (16, 76, 76)   ->  (32, 40, 40)
    # conv3 (32, 40, 40)   ->  (64, 22, 22)
    # conv4 (64, 22, 22)   ->  (128, 12, 12)
    # conv5 (128, 12, 12)    ->  (256, 8, 8)
    # conv6 (256, 8, 8)    ->  (512, 4, 4)
    # fc    (256*4*4)       ->  (n_z, 1, 1)
    def __init__(self):
        super(enc, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, n_ef, 4, 2, 2, bias=False),
            nn.BatchNorm2d(n_ef),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(n_ef, n_ef*2, 4, 2, 3, bias=False),
            nn.BatchNorm2d(n_ef*2),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv3 = nn.Sequential(
            nn.Conv2d(n_ef*2, n_ef*4, 4, 2, 3, bias=False),
            nn.BatchNorm2d(n_ef*4),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv4 = nn.Sequential(
            nn.Conv2d(n_ef*4, n_ef*8, 4, 2, 2, bias=False),
            nn.BatchNorm2d(n_ef*8),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv5 = nn.Sequential(
            nn.Conv2d(n_ef*8, n_ef*16, 4, 2, 3, bias=False),
            nn.BatchNorm2d(n_ef*16),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv6 = nn.Sequential(
            nn.Conv2d(n_ef*16, n_ef*32, 4, 2, 2, bias=False),
            nn.BatchNorm2d(n_ef*32),
            nn.LeakyReLU(0.2, inplace=True))

        self.mu = nn.Linear(n_ef*32*5*5, n_z, bias=False)
        self.logvar = nn.Linear(n_ef*32*5*5, n_z, bias=False)

    def forward(self, x):
        # calculating the logvariance ensurees that the std is positive
        # since exp(logvar) = std^2 > 0
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x).view(-1, n_ef*32*5*5)
        mu = self.mu(x).view(-1, n_z, 1, 1)
        logvar = self.logvar(x).view(-1, n_z, 1, 1)
        return mu, logvar


class dec(nn.Module):
    # conv1 (n_z, 1, 1)   ->  (256, 5, 5)
    # conv2 (256, 5, 5)   ->  (128, 12, 12)
    # conv3 (128, 12, 12)   ->  (64, 22, 22)
    # conv4 (64, 22, 22)  ->  (32, 40 ,40)
    # conv5 (32, 40, 40)  ->  (16, 76, 76)
    # conv6 (16, 76, 76)  ->  (1, 150, 150)
    def __init__(self):
        super(dec, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(n_z, n_gf*32, 5, 1, 0, bias=False),
            nn.BatchNorm2d(n_gf*32),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(n_gf*32, n_gf*16, 4, 2, 2, bias=False),
            nn.BatchNorm2d(n_gf*16),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(n_gf*16, n_gf*8, 4, 2, 3, bias=False),
            nn.BatchNorm2d(n_gf*8),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(n_gf*8, n_gf*4, 4, 2, 2, bias=False),
            nn.BatchNorm2d(n_gf*4),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(n_gf*4, n_gf*2, 4, 2, 3, bias=False),
            nn.BatchNorm2d(n_gf*2),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(n_gf*2, n_gf, 4, 2, 3, bias=False),
            nn.BatchNorm2d(n_gf),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv7 = nn.Sequential(
            nn.ConvTranspose2d(n_gf, 1, 4, 2, 2, bias=False),
            nn.Sigmoid())

    def forward(self, z):
        x = self.conv1(z)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x).view(-1, 1, 150, 150)
        return x

    def backprop(self, y_pred, y ,loss, retain_graph=True):
        """
        Pass a tuple of tuples with each tuple of form (X, y, loss) 
        """
        L = loss(y_pred, y)
        L.backward(retain_graph=retain_graph)
        return L


class disc(nn.Module):
    def __init__(self):
        super(disc, self).__init__()
        # Conv2D(in_channels, out_channels, kernel size, stride, padding)

        # conv1 (1, 150, 150)   ->  (16, 76, 76)
        # conv2 (16, 76, 76)    ->  (32, 40, 40)
        # conv3 (32, 40, 40)    ->  (64, 22, 22)
        # conv4 (64, 22, 22)    ->  (128, 12, 12)
        # conv5 (128, 12, 12)   ->  (256, 8, 8)
        # conv6 (256, 8, 8)     ->  (512, 5, 5)
        # conv7 (512, 4, 4)     ->  (1, 1, 1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, n_df, 4, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(n_df, n_df*2, 4, 2, 3, bias=False),
            nn.BatchNorm2d(n_df*2),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv3 = nn.Sequential(
            nn.Conv2d(n_df*2, n_df*4, 4, 2, 3, bias=False),
            nn.BatchNorm2d(n_df*4),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv4 = nn.Sequential(
            nn.Conv2d(n_df*4, n_df*8, 4, 2, 2, bias=False),
            nn.BatchNorm2d(n_df*8),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv5 = nn.Sequential(
            nn.Conv2d(n_df*8, n_df*16, 4, 2, 3, bias=False),
            nn.BatchNorm2d(n_df*16),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(n_df*16, n_df*32, 4, 2, 2, bias=False),
            nn.BatchNorm2d(n_df*32),
            nn.LeakyReLU(0.2, inplace=True))

        self.conv7 = nn.Sequential(
            nn.Conv2d(n_df*32, 1, 5, 1, 0, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        D_l = x.clone()
        x = self.conv5(x)
        x = self.conv6(x)
        y_pred = self.conv7(x).view(-1)
        return y_pred, D_l

    def backprop(self, X, y, loss, retain_graph=False):
        """
        Pass a tuple of tuples with each tuple of form (X, y, loss) 
        """
        y_pred = self.forward(X)[0].view(-1)
        L = loss(y_pred, y)
        L.backward(retain_graph=retain_graph)
        return L


class I(nn.Module):
    def __init__(self):
        super(I, self).__init__()
        # Conv2D(in_channels, out_channels, kernel size, stride, padding)

        # conv1 (1, 150, 150)  ->  (32, 75, 75)
        # conv2 (32, 64, 64)   ->  (64, 36, 36)
        # conv3 (64, 4, 4)     ->  (128, 18, 18)
        # conv4 (128, 4, 4)    ->  (256, 8, 8)
        # conv5 (256, 8, 8)    ->  (3, 1, 1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 11, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2))

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 24, 3, 1, 1),
            nn.BatchNorm2d(24),
            nn.ReLU())

        self.conv4 = nn.Sequential(
            nn.Conv2d(24, 24, 3, 1, 1),
            nn.BatchNorm2d(24),
            nn.ReLU())

        self.conv5 = nn.Sequential(
            nn.Conv2d(24, 16, 3, 1, 1),
            nn.BatchNorm2d(16,),
            nn.ReLU(),
            nn.MaxPool2d(5, stride=4))

        self.fid_layer = torch.zeros(256)

        # 8192 -> 2048
        # 2048 -> 512
        # 512  -> 512
        # 512  -> 3
        self.linear1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout())

        self.linear2 = nn.Sequential(
            nn.Linear(256, 2),
            nn.Softmax(dim=1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, 256)
        self.fid_layer = x
        x = self.linear1(x)
        x = self.linear2(x)

        return x

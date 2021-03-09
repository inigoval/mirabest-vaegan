import torch
import torchvision
import torch
import numpy as np
import torchvision.transforms as T

from datasets import MiraBest_full, MB_nohybrids, MBFRConfident
from paths import Path_Handler

paths = Path_Handler()
path_dict = paths._dict()


class Circle_Crop(torch.nn.Module):
    """
    Set all values outside largest possible circle that fits inside image to 0
    """

    def __init__(self):
        super().__init__()

    def forward(self, img):
        """
        !!! Support for multiple channels not implemented yet !!!
        """
        H, W, C = img.shape[-1], img.shape[-2], img.shape[-3]
        assert H == W
        x = torch.arange(W, dtype=torch.float).repeat(H, 1)
        x = (x-74.5)/74.5
        y = torch.transpose(x, 0, 1)
        r = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2))
        r = r/torch.max(r)
        r[r < 0.5] = -1
        r[r == 0.5] = -1
        r[r != -1] = 0
        r = torch.pow(r, 2).view(1, H, W)
        assert r.shape == img.shape
        img = torch.mul(r, img)
        return img

transform = torchvision.transforms.Compose([
                         T.RandomRotation(180),
                         T.ToTensor(),
                         Circle_Crop()])

class Data_Agent():
    def __init__(self, dataset, path=path_dict['data'], transform=transform, batch_size=32, seed=69, download=False):
        self.batch_size = batch_size
        self.seed = seed
        self.transform = transform
        self.path = path

        self.train = dataset(path, train=True, transform=transform, download=download)
        self.test = dataset(path, train=False, transform=transform, download=download)
        self.n_test = len(self.test)
 
    def subset(self, fraction):
        self.fraction=fraction
        self.set_seed()

        length = len(self.train)
        idx = np.arange(length)
        subset_idx = np.random.choice(idx, size=int(fraction*length))
        subset = torch.utils.data.Subset(self.train, subset_idx)
        self.train = subset

    def set_labels(self, label):
        self.choose_label(self.train, label)
        self.choose_label(self.test, label)
        self.n_test = len(self.test)

    def set_seed(self):
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)

    def load(self):
        self.set_seed()
        train_loader = torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(self.test, batch_size=self.batch_size, shuffle=True)
        return train_loader, test_loader

    def fid_dset(self, size=10000):
        all_data = torch.utils.data.ConcatDataset((self.train, self.test))
        loader = torch.utils.data.DataLoader(all_data, batch_size=len(all_data), shuffle=True)

        # Complete enough cycles to have 'size' number of samples 
        n_cycles = int(size/len(all_data))
        X_fid, y_fid = torch.FloatTensor(), torch.LongTensor()
        for i in np.arange(n_cycles):
            for data in loader:
                X, y = data
                X_fid = torch.cat((X_fid, X), 0)
                y_fid = torch.cat((y_fid, y), 0)

        self.X_fid = X_fid.cpu()
        self.y_fid = y_fid.cpu()

    @staticmethod
    def choose_label(dataset, label):
        """
        Reduce input dataset to only contain given label
        """
        data_array = dataset.data
        target_list = dataset.targets
        targets = np.asarray(target_list)
        label_idx = np.argwhere(targets == label)
        targets = targets[label_idx]
        data_array = data_array[label_idx]
        dataset.data = data_array
        dataset.targets = targets

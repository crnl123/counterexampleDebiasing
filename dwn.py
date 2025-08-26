import random
from abc import ABC, abstractmethod
import numpy as np
import torchvision
from torchvision.transforms import v2
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataset import _T_co


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.drop1 = nn.Dropout2d(0.2)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2_1 = nn.Conv2d(32, 64, 3)
        self.drop2_1 = nn.Dropout2d(0.2)
        self.conv2_2 = nn.Conv2d(64, 64, 3)
        self.drop2_2 = nn.Dropout2d(0.2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3_1 = nn.Conv2d(64, 128, 3)
        self.drop3_1 = nn.Dropout2d(0.2)
        self.conv3_2 = nn.Conv2d(128, 128, 3)
        self.drop3_2 = nn.Dropout2d(0.2)
        # self.pool3 = nn.AdaptiveAvgPool2d(10)  # 128,1,1 so unneeded

        self.dense = nn.Linear(128, 10)
        pass

    def forward(self, x):
        x = self.conv1(x)
        x = self.drop1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.drop2_1(x)
        x = F.relu(x)
        x = self.conv2_2(x)
        x = self.drop2_2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.drop3_1(x)
        x = F.relu(x)
        x = self.conv3_2(x)
        x = self.drop3_2(x)
        x = F.relu(x)
        # x = self.pool3(x)  # 128,1,1 so unneeded

        x = x.squeeze(-1).squeeze(-1)

        x = self.dense(x)
        out = F.softmax(x, dim=1)
        return out


class UnderCoverKeyEncoder(nn.Module):
    """
    A wierd design
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(6, 32, 3, padding='same')  # 3 channel from each image
        self.norm1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding='same')
        self.norm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding='same')
        self.norm3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 64, 3, padding='same')
        self.norm4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(128, 32, 3, padding='same')  # Takes 2+4
        self.norm5 = nn.BatchNorm2d(32)

        self.convf = nn.Conv2d(64, 3, 3, padding='same')  # Takes 1+5

    def forward(self, x, z):
        z3 = torch.cat((z, z, z), 1)
        h = torch.cat((x, z3), 1)  # 0=batch, 1=channel, 2=height, 3=width

        h1 = self.conv1(h)
        h1 = self.norm1(h1)
        h1 = F.leaky_relu(h1)

        h2 = self.conv2(h1)
        h2 = self.norm2(h2)
        h2 = F.leaky_relu(h2)

        h3 = self.conv3(h2)
        h3 = self.norm3(h3)
        h3 = F.leaky_relu(h3)

        h4 = self.conv4(h3)
        h4 = self.norm4(h4)
        h4 = F.leaky_relu(h4)

        h5 = torch.cat((h2, h4), 1)
        h5 = self.conv5(h5)
        h5 = self.norm5(h5)
        h5 = F.leaky_relu(h5)

        hf = torch.cat((h1, h5), 1)
        out = self.convf(hf)
        return out


class UnderCoverKeyDecoder(nn.Module):
    """
    A wierd design 2
    """

    def __init__(self):
        super().__init__()
        self.conv6 = nn.Conv2d(3, 32, 3, padding='same')
        self.norm6 = nn.BatchNorm2d(32)

        self.conv7 = nn.Conv2d(32, 64, 3, padding='same')
        self.norm7 = nn.BatchNorm2d(64)

        self.conv8 = nn.Conv2d(64, 128, 3, padding='same')
        self.norm8 = nn.BatchNorm2d(128)

        self.conv9 = nn.Conv2d(128, 64, 3, padding='same')
        self.norm9 = nn.BatchNorm2d(64)

        self.conv10 = nn.Conv2d(128, 32, 3, padding='same')  # Takes 7+9
        self.norm10 = nn.BatchNorm2d(32)

        self.convxp = nn.Conv2d(64, 3, 3, padding='same')
        self.convzp = nn.Conv2d(64, 3, 3, padding='same')

    def forward(self, x):
        h6 = self.conv6(x)
        h6 = self.norm6(h6)
        h6 = F.leaky_relu(h6)

        h7 = self.conv7(h6)
        h7 = self.norm7(h7)
        h7 = F.leaky_relu(h7)

        h8 = self.conv8(h7)
        h8 = self.norm8(h8)
        h8 = F.leaky_relu(h8)

        h9 = self.conv9(h8)
        h9 = self.norm9(h9)
        h9 = F.leaky_relu(h9)

        h10 = torch.cat((h7, h9), 1)
        h10 = self.conv10(h10)
        h10 = self.norm10(h10)
        h10 = F.leaky_relu(h10)

        hf = torch.cat((h6, h10), 1)

        xp = self.convxp(hf)
        zp = self.convzp(hf)

        return xp, zp


class UnderCoverModel(nn.Module):
    def __init__(self, vanilla=True):
        super().__init__()
        if vanilla:
            self.keyEncoder = UnderCoverKeyEncoder()
            self.keyDecoder = UnderCoverKeyDecoder()
        else:
            raise NotImplementedError('Custom version is to be implemented.')

        self.xClassifier = Classifier()
        self.wClassifier = Classifier()

    def forward(self, x, z):
        xh = self.keyEncoder(x, z)
        xp, zp = self.keyDecoder(xh)

        w = xh - x  # Global bias omitted according to original code

        pred_x, pred_xh, pred_xp = self.xClassifier(x), self.xClassifier(xh), self.xClassifier(xp)
        pred_w = self.wClassifier(w)

        return xh, xp, zp, pred_xh, pred_xp, pred_x, pred_w


class MyKeyEncoder(nn.Module):
    """
    My attempt
    """

    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def forward(self, x1, x2):
        raise NotImplementedError


class MyKeyDecoder(nn.Module):
    """
    My attempt 2
    """

    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def forward(self, x1, x2):
        raise NotImplementedError


class ParallelDataset(torch.utils.data.Dataset, ABC):
    def __init__(self, x_set, z_set):
        self.x_set = x_set
        self.z_set = z_set
        self.length = min(len(self.x_set), len(self.z_set))
        self.x_indices = None
        self.z_indices = None
        self.set_xz()

    def __len__(self):
        return self.length

    def __getitem__(self, index) -> _T_co:
        x_index = self.x_indices[index]
        z_index = self.z_indices[index]

        x, yx = self.x_set[x_index]
        z, yz = self.z_set[z_index]
        return x, yx, z, yz

    @abstractmethod
    def set_xz(self):
        raise NotImplementedError


class RandomParallelDataset(ParallelDataset):
    def __init__(self, x_set, z_set):
        super().__init__(x_set, z_set)

    def set_xz(self):
        self.x_indices = list(range(self.length))
        random.shuffle(self.x_indices)
        self.z_indices = list(range(self.length))
        random.shuffle(self.z_indices)


class OrderedParallelDataset(ParallelDataset):
    def __init__(self, x_set, z_set):
        super().__init__(x_set, z_set)

    def set_xz(self):
        # Get yx, yz arrays
        yxs = []
        for _, yx in self.x_set:
            yxs.append(yx)
        yxs = np.array(yxs)

        yzs = []
        for _, yz in self.z_set:
            yzs.append(yz)
        yzs = np.array(yzs)

        x_acc = np.empty((0,), dtype='uint8')
        z_acc = np.empty((0,), dtype='uint8')

        for i in range(10):
            # Get per-class index list. Iterate over i: 0-9, return list of indices where i == y
            xi = np.where(yxs == i)[0]
            zi = np.where(yzs == i)[0]

            x_acc = np.concatenate((x_acc, xi))

            # Randomly choose from yz to match yx length
            np.random.shuffle(zi)
            z_acc = np.concatenate((z_acc, zi[:5000]))

        self.x_indices = list(x_acc)
        self.z_indices = list(z_acc)


class DWNLoss(nn.Module):
    def __init__(self, loss_weights=(20., 1., 1., 0.03, 0.03, 0.03, 5.)):
        super().__init__()
        self.a_xh_x, self.a_xp_x, self.a_zp_z, self.a_xh_yx, self.a_xp_yx, self.a_x_yx, self.a_zp_yz = loss_weights
        self.factor = self.a_xh_x + self.a_xp_x + self.a_zp_z + self.a_xh_yx + self.a_xp_yx + self.a_x_yx + self.a_zp_yz

        self.mae = nn.L1Loss()
        self.cel = nn.CrossEntropyLoss()

    def forward(self, ins, outs, ys):
        x, z = ins
        xh, xp, zp, pred_xh, pred_xp, pred_x, pred_w = outs
        yx, yz = ys

        xh_x = self.mae(xh, x)
        xp_x = self.mae(xp, x)
        zp_z = self.mae(zp, torch.cat((z, z, z), 1))

        xh_yx = self.cel(pred_xh, yx)
        xp_yx = self.cel(pred_xp, yx)
        x_yx = self.cel(pred_x, yx)
        zp_yz = self.cel(pred_w, yz)

        losses = (xh_x, xp_x, zp_z, xh_yx, xp_yx, x_yx, zp_yz)
        loss = self.normalize_and_sum(losses)

        return loss

    def normalize_and_sum(self, losses):
        xh_x, xp_x, zp_z, xh_yx, xp_yx, x_yx, zp_yz = losses

        xh_x *= self.a_xh_x / self.factor
        xp_x *= self.a_xp_x / self.factor
        zp_z *= self.a_zp_z / self.factor

        xh_yx *= self.a_xh_yx / self.factor
        xp_yx *= self.a_xp_yx / self.factor
        x_yx *= self.a_x_yx / self.factor
        zp_yz *= self.a_zp_yz / self.factor

        return xh_x + xp_x + zp_z + xh_yx + xp_yx + x_yx + zp_yz


def load_data(train=True):
    fashion_transform = torch.nn.Sequential(
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.286], std=[1]),
        v2.Pad(padding=2)
    )

    cifar_transform = torch.nn.Sequential(
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.491, 0.482, 0.447], std=[1]),
        # v2.Pad(padding=2)
    )

    fashion = torchvision.datasets.FashionMNIST(
        root='datasets/fashionMNIST',
        train=train,
        download=True,
        transform=fashion_transform)

    cifar = torchvision.datasets.CIFAR10(
        root='datasets/CIFAR10',
        train=train,
        download=True,
        transform=cifar_transform)

    return fashion, cifar

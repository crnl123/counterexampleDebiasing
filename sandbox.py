import torch
import pickle
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import v2
from torch.utils import data
import dwn
import utils
import numpy as np
import matplotlib.pyplot as plt

# fashion, cifar = dwn.load_data()
# set = dwn.RandomParallelDataset(cifar, fashion)

# f = open('datasets/watermark/debiased_quarters.pickle', 'rb')
# mix_x, mix_y = pickle.load(f)
# set = utils.XYDataset(mix_x, mix_y)
# train_loader = data.DataLoader(
#     set,
#     batch_size=1,
#     num_workers=0,
#     shuffle=True)
#
unnorm = v2.Normalize(mean=[-0.491, -0.482, -0.447], std=[1])
#
f = open('datasets/watermark/dataset_1.0.pickle', 'rb')
x, xh, w, yx, yz = pickle.load(f)
f.close()

# a = utils.CutMix('random_quarters', beta=1)(x[0], xh[10000])
#
#
# utils.show(torch.permute(unnorm(a), (1, 2, 0)))

# utils.show(torch.permute(unnorm(xh[0]), (1, 2, 0)))
utils.show(torch.permute(unnorm(x[10000]), (1, 2, 0)))

# utils.get_mean_std(
#     train_loader=data.DataLoader(
#         cifar,
#         batch_size=1,
#         num_workers=0,
#         shuffle=True))

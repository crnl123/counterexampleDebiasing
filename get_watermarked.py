import pickle
import torch
from torch.utils import data
from tqdm import tqdm
import dwn
import numpy as np

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train = True
data_path = 'datasets/watermark/dataset.pickle' if train else 'datasets/watermark/dataset_test.pickle'
weight_path = 'weights/dwn.pth'
batch_size = 1024

# Load model and data
print(f'=Extracting watermark with {device}=')

fashion, cifar = dwn.load_data(train=train)

ordered_parallel = dwn.OrderedParallelDataset(cifar, fashion)

train_loader = data.DataLoader(
    ordered_parallel,
    batch_size=batch_size,
    num_workers=0,
    shuffle=False)

model = dwn.UnderCoverModel()
model.to(device)
model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
model.eval()

# Accumulators
x_acc = None
xh_acc = None
w_acc = None
yx_acc = None
yz_acc = None

# Predict
with torch.no_grad():
    for i, (x, yx, z, yz) in enumerate(tqdm(train_loader)):
        # Run
        x = x.to(device)
        z = z.to(device)

        xh, xp, zp, _, _, _, _ = model(x, z)
        w = xh - x

        x = x.cpu()
        xh = xh.cpu()
        w = w.cpu()
        yx = yx.cpu()
        yz = yz.cpu()

        # Accumulate
        if x_acc is None:
            x_acc = x
        else:
            x_acc = torch.concatenate((x_acc, x))

        if xh_acc is None:
            xh_acc = xh
        else:
            xh_acc = torch.concatenate((xh_acc, xh))

        if w_acc is None:
            w_acc = w
        else:
            w_acc = torch.concatenate((w_acc, w))

        if yx_acc is None:
            yx_acc = yx
        else:
            yx_acc = torch.concatenate((yx_acc, yx))

        if yz_acc is None:
            yz_acc = yz
        else:
            yz_acc = torch.concatenate((yz_acc, yz))

print('Generation completed')

# Save
f = open(data_path, 'wb')
pickle.dump((x_acc, xh_acc, w_acc, yx_acc, yz_acc), f)

print('Files saved')

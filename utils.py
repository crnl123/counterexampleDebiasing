import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.transforms import v2
from PIL import Image
import torch.nn.functional as f

def get_mean_std(train_loader):
    imgs = None
    for _, batch in enumerate(tqdm(train_loader)):
        image_batch = batch[0]
        if imgs is None:
            imgs = image_batch.cpu()
        else:
            imgs = torch.cat([imgs, image_batch.cpu()], dim=0)
    imgs = imgs.numpy()

    # calculate mean over each channel (r,g,b)
    mean_r = imgs[:, 0, :, :].mean()
    try:
        mean_g = imgs[:, 1, :, :].mean()
        mean_b = imgs[:, 2, :, :].mean()
        print(f'3 channel mean: {round(mean_r, 3)}, {round(mean_g, 3)}, {round(mean_b, 3)}')
    except IndexError:
        print(f'Grayscale mean: {round(mean_r, 3)}')

    # calculate std over each channel (r,g,b)
    std_r = imgs[:, 0, :, :].std()
    try:
        std_g = imgs[:, 1, :, :].std()
        std_b = imgs[:, 2, :, :].std()
        print(f'3 channel std: {round(std_r, 3)}, {round(std_g, 3)}, {round(std_b, 3)}')
    except IndexError:
        print(f'Grayscale std: {round(std_r, 3)}')


def show(img):
    plt.figure(figsize=(20, 20))
    plt.imshow(img)
    plt.axis('off')
    plt.show()


class XYDataset(torch.utils.data.Dataset):
    def __init__(self, x_set, y_set, to_one_hot=False):
        super().__init__()
        self.x_set = x_set
        self.y_set = y_set
        if len(x_set) != len(y_set):
            raise Exception(f'X, Y size mismatch: X={len(x_set)}, Y={len(y_set)}')
        self.length = len(self.x_set)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.x_set[index], self.y_set[index]

    def to_one_hot(self):
        self.y_set = f.one_hot(self.y_set.to(torch.int64), 10).to(torch.float32)


class CutMix(torch.nn.Module):
    def __init__(self, box_mode="random_halves", beta=1):
        super().__init__()
        self.box_mode = box_mode
        self.beta = torch.distributions.Beta(beta, beta)
        self.label_mixing = is_label_mixing(box_mode)
        if self.label_mixing:
            self.box_mode = box_mode[7:]

    def forward(self, xi, xhi):
        assert xi.size() == xhi.size()
        half_x = xi.size()[-1] // 2
        end_x = xi.size()[-1]
        half_y = xi.size()[-2] // 2
        end_y = xi.size()[-2]
        half_square = int(half_x * end_x ** 0.5)
        rand = 0
        if self.box_mode == "id":
            return xi.clone()
        if self.box_mode == "random_halves":
            rand = random.randint(1, 4)
        if self.box_mode == "random_quarters":
            rand = random.randint(5, 8)

        if self.box_mode == "marked_bottom" or rand == 1:
            x1, x2, y1, y2 = 0, end_x, half_y, end_y
        elif self.box_mode == "marked_right" or rand == 2:
            x1, x2, y1, y2 = 0, half_x, 0, end_y
        elif self.box_mode == "marked_top" or rand == 3:
            x1, x2, y1, y2 = 0, end_x, 0, half_y
        elif self.box_mode == "marked_left" or rand == 4:
            x1, x2, y1, y2 = half_x, end_x, 0, end_y

        elif self.box_mode == "marked_top_left" or rand == 5:
            x1, x2, y1, y2 = 0, half_x, 0, half_y
        elif self.box_mode == "marked_top_right" or rand == 6:
            x1, x2, y1, y2 = half_x, end_x, 0, half_y
        elif self.box_mode == "marked_bottom_left" or rand == 7:
            x1, x2, y1, y2 = 0, half_x, half_y, end_y
        elif self.box_mode == "marked_bottom_right" or rand == 8:
            x1, x2, y1, y2 = half_x, end_x, half_y, end_y

        elif self.box_mode == "random_box":
            rand_x = torch.randint(0, end_x + 1, ())
            rand_y = torch.randint(0, end_y + 1, ())
            rand_x_size = self.beta.sample(()) * end_x // 2
            rand_y_size = self.beta.sample(()) * end_y // 2

            x1 = int(torch.clamp(rand_x - rand_x_size, min=0))
            y1 = int(torch.clamp(rand_y - rand_y_size, min=0))
            x2 = int(torch.clamp(rand_x + rand_x_size, max=end_x))
            y2 = int(torch.clamp(rand_y + rand_y_size, max=end_y))

        elif self.box_mode == "vanilla":
            x1 = random.randint(0, end_x)
            y1 = random.randint(0, end_y)

            frac = 0.25

            dx = end_x * (1 - frac) ** 0.5
            dy = end_y * (1 - frac) ** 0.5

            x2 = x1 + dx
            y2 = y1 + dy

        else:
            raise Exception(f"Bad arguments: unrecognized box_mode {self.box_mode}")
        output = xi.clone()
        output[..., y1:y2, x1:x2] = xhi[..., y1:y2, x1:x2]
        width = y2-y1
        height = x2-x1
        hsize = width*height
        total = end_x * end_y
        frac = hsize / total
        # show(unnorm(output))
        if not self.label_mixing:
            return output, 0
        else:
            return output, frac


def im_save(im, filepath):
    im = Image.fromarray(im)
    im.save(filepath)


def unnorm(img):
    t = v2.Normalize(mean=[-0.491, -0.482, -0.447], std=[1])
    return torch.permute(t(img), (1, 2, 0))


def is_label_mixing(box_mode):
    return box_mode[0:7] == 'CUTMIX_'

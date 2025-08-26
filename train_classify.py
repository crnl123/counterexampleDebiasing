import pickle
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
import dwn
from torchvision.transforms import v2
from torch.utils import data
import utils


def trim(x, xh, y, clean_size=500, class_size=5000):
    """
    Divides classes into (clean_size) clean samples and (class_size-clean_size) marked samples with no overlap
    :param x: Ordered clean set of size NxCxHxW
    :param xh: Ordered marked set of size NxCxHxW
    :param y: Ordered label set of size N
    :param clean_size: Number of clean samples per class, default=500
    :param class_size: Number of original samples per class, default=5000
    :return: List of dictionaries containing xh, yxh, x, yx
    """
    if clean_size > class_size:
        raise Exception(
            f"Bad arguments: clean_size must not be greater than class_size, but got {clean_size}, {class_size}.")
    acc = []
    for class_i in range(10):
        start_i = class_i * class_size
        end_i = start_i + class_size

        acc.append({"xh": xh[start_i + clean_size:end_i],
                    "yxh": y[start_i + clean_size:end_i],
                    "x": x[start_i:start_i + clean_size],
                    "yx": y[start_i:start_i + clean_size]})
    return acc


def make_sets(classes):
    x = torch.empty((0,))
    yx = torch.empty((0,))
    xh = torch.empty((0,))
    yxh = torch.empty((0,))

    for c in classes:
        x = torch.cat((x, c["x"]))
        yx = torch.cat((yx, c["yx"]))
        xh = torch.cat((xh, c["xh"]))
        yxh = torch.cat((yxh, c["yxh"]))

    clean_set = utils.XYDataset(x, yx)
    marked_set = utils.XYDataset(xh, yxh)
    return clean_set, marked_set


def mix(size, classes, mix_func):
    """

    :param size: How big the mixed set should be for each x class
    :param classes: List of dictionaries containing xh, yxh, x, yx
    :param mix_func: Callable function to mix x and xh
    :return:
    """
    # x_acc, y_acc = torch.empty((0,)), torch.empty((0,))  # Unused b.c. this method is slow
    x_acc, y_acc = [], []
    for c in range(len(classes)):
        print(f'Class {c}')
        x = classes[c]["x"]
        yx = classes[c]["yx"]
        other_cs = classes[:c] + classes[c + 1:]

        xh = torch.empty((0,))
        for c_ in other_cs:
            xh = torch.cat((xh, c_["xh"]))

        rand_xhis = torch.randperm(len(xh))  # Random list of indices
        rand_xis = torch.randperm(len(x))
        for i in tqdm(range(size), leave=False):
            rand_i = rand_xhis[i % len(rand_xhis)]
            xhi = xh[rand_i]  # Modulo over shuffled xh indices and choose sample

            rand_i = rand_xis[i % len(rand_xis)]
            xi = x[rand_i]  # Modulo over shuffled x indices and choose sample
            yi = yx[rand_i:rand_i + 1]

            mix_i = mix_func(xi, xhi)

            # x_acc = torch.cat((x_acc, torch.unsqueeze(mix_i, 0)))
            # y_acc = torch.cat((y_acc, yi))
            x_acc.append(mix_i)
            y_acc.append(yi)
    return torch.stack(x_acc), torch.cat(y_acc)


def train(model, train_loader, test_loaders, optimizer, loss_func, device=torch.device('cpu'), use_tqdm=True):
    model.train()
    for i, (x, y) in enumerate(tqdm(train_loader, leave=False, disable=not use_tqdm)):
        x = x.to(device)
        y = y.to(device).to(torch.int64)

        o = model(x)

        loss = loss_func(o, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_losses = []
    for test_loader in test_loaders:
        test_losses.append(get_loss(model, test_loader, loss_func, device))

    return test_losses


def get_loss(model, train_loader, loss_func, device=torch.device('cpu')):
    loss = 0.
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        o = model(x)

        loss += float(loss_func(o, y).detach().cpu())
    loss /= len(train_loader.dataset)
    return loss


def get_mixed(classes, mixed_path, box_mode, mix_size=4500 * 9):
    try:
        f = open(mixed_path, 'rb')
        train_x, train_y = pickle.load(f)
    except FileNotFoundError:
        train_x, train_y = mix(mix_size, classes, utils.CutMix(box_mode))
        f = open(mixed_path, 'wb')
        pickle.dump((train_x, train_y), f)
    return train_x, train_y


def get_test_loaders(x_test, xh_test, w_test, y_test, batch_size):
    clean_test = utils.XYDataset(x_test, y_test)
    marked_test = utils.XYDataset(xh_test, y_test)
    watermark_test = utils.XYDataset(w_test, y_test)

    clean_test_loader = data.DataLoader(
        clean_test,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True)

    marked_test_loader = data.DataLoader(
        marked_test,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True)

    watermark_test_loader = data.DataLoader(
        watermark_test,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True)
    return clean_test_loader, marked_test_loader, watermark_test_loader


def main():
    print(f'=Training classifier with {device}=')
    f = open(dataset_path, 'rb')
    x, xh, w, y, _ = pickle.load(f)  # Normalized data, ordered as 10 classes x 5000 samples
    f.close()

    f = open(test_path, 'rb')
    x_test, xh_test, w_test, y_test, _ = pickle.load(f)  # Normalized data, ordered as 10 classes x 1000 samples
    f.close()

    if mixed_debias_training:
        classes = trim(x, xh, y)
        train_x, train_y = get_mixed(classes, train_path, box_mode)  # When training with mixed data

        print('Raw data loaded - debiased training')

        clean_train, marked_train = make_sets(classes)
        mixed_train = utils.XYDataset(train_x, train_y)
        whole_train = data.ConcatDataset((clean_train, marked_train, mixed_train))
    else:  # Watermarked only
        whole_train = utils.XYDataset(xh, y)
        print('Raw data loaded - watermark training')

    train_loader = data.DataLoader(
        whole_train,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True)

    test_loaders = get_test_loaders(x_test, xh_test, w_test, y_test, batch_size)

    print("Dataloaders ready")

    model = dwn.Classifier()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters())

    loss_func = nn.CrossEntropyLoss()

    print("Model Ready")

    best = 99999.

    for epoch in range(epochs):
        saving = False
        loss = train(model, train_loader, test_loaders, optimizer, loss_func, device)
        if loss[test_loss_type] < best:
            best = loss[test_loss_type]
            torch.save(model.state_dict(), weight_path)
            saving = True
        print(f'Epoch: {epoch}, loss: {loss} {"Saving" if saving else ""}')


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = 'datasets/watermark/dataset.pickle'
    test_path = 'datasets/watermark/dataset_test.pickle'
    weight_path = 'weights/classifier_watermark.pth'

    mixed_debias_training = True
    if mixed_debias_training:
        weight_path = 'weights/classifier_quarters_P-match.pth'
        train_path = 'datasets/watermark/quarters_P-match.pickle'
        # train_path = 'datasets/watermark/debiased_quarters.pickle'
        box_mode = "random_quarters"

    test_loss_types = {
        'clean': 0,
        'marked': 1,
        'watermark': 2
    }
    test_loss_type = test_loss_types['clean']
    epochs = 15
    batch_size = 128

    main()

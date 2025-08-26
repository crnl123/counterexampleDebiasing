import pickle
import torch
from torch.utils import data
import dwn
import utils
from torcheval.metrics import MulticlassAccuracy
from train_classify import get_test_loaders
from torch import nn
import numpy as np


def get_loss(model, train_loader, loss_func, device=torch.device("cpu")):
    """
    duplicated b.c. device is global
    :param model:
    :param train_loader:
    :param loss_func:
    :return:
    """
    loss = 0.
    metric = MulticlassAccuracy()
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        o = model(x)

        metric.update(o, y)
        loss += float(loss_func(o, y).detach().cpu())
    loss /= len(train_loader.dataset)
    return loss, float(metric.compute().detach().cpu())


def test(weight_path, test_loaders, device=torch.device('cpu')):
    model = dwn.Classifier()
    model.to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval()
    print(f'Model ready: {weight_path}')

    loss_func = nn.CrossEntropyLoss()

    test_losses = []
    for test_loader in test_loaders:
        test_losses.append(get_loss(model, test_loader, loss_func, device))

    return test_losses


def main():
    f = open(test_path, 'rb')
    x_test, xh_test, w_test, y_test, _ = pickle.load(f)  # Normalized data, ordered as 10 classes x 1000 samples
    f.close()

    test_loaders = get_test_loaders(x_test, xh_test, w_test, y_test, batch_size)

    print(test(p_match_weight_path, test_loaders, device))



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_match_weight_path = 'weights/classifier_watermark.pth'
    test_path = 'datasets/watermark/dataset_0.75_test.pickle'
    batch_size = 128
    main()

import pickle
import torch
from torch import nn
from tqdm import tqdm

import dwn
import utils
from train_classify import get_test_loaders
import numpy as np


def get_preds(model, train_loader, device=torch.device("cpu")):
    """
    duplicated b.c. device is global
    :param device:
    :param model:
    :param train_loader:
    :param loss_func:
    :return:
    """
    pos_acc = []
    neg_acc = []
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        o = model(x)

        pos = torch.argmax(o, dim=1).detach().cpu().numpy() == y.detach().cpu().numpy()
        neg = torch.argmax(o, dim=1).detach().cpu().numpy() != y.detach().cpu().numpy()
        pos_acc.append(x.detach().cpu().numpy()[pos])
        neg_acc.append(x.detach().cpu().numpy()[neg])
    return np.concatenate(pos_acc), np.concatenate(neg_acc)


def test(weight_path, test_loaders, device=torch.device('cpu')):
    model = dwn.Classifier()
    model.to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval()
    print(f'Model ready: {weight_path}')

    loss_func = nn.CrossEntropyLoss()

    preds = []
    for test_loader in test_loaders:
        preds.append(get_preds(model, test_loader, device))

    return preds


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f = open('datasets/watermark/dataset_test.pickle', 'rb')
    x_test, xh_test, w_test, y_test, _ = pickle.load(f)  # Normalized data, ordered as 10 classes x 1000 samples
    f.close()

    batch_size = 128
    test_loaders = get_test_loaders(x_test, xh_test, w_test, y_test, batch_size)  # clean, watermarked, watermark

    clean_positives, clean_negatives = test('weights/classifier_quarters_P-match.pth', test_loaders, device)[0]

    for i, img in tqdm(enumerate(clean_positives)):
        temp = utils.unnorm(torch.Tensor(img)).numpy()
        temp = (temp * 255).astype(np.uint8)
        utils.im_save(temp, f'examples/clean_positives/positive_{i}.png')

    for i, img in tqdm(enumerate(clean_negatives)):
        temp = utils.unnorm(torch.Tensor(img)).numpy()
        temp = (temp * 255).astype(np.uint8)
        utils.im_save(temp, f'examples/clean_negatives/negative_{i}.png')

    clean_positives, clean_negatives = test('weights/classifier_quarters_P-match.pth', test_loaders, device)[1]

    for i, img in tqdm(enumerate(clean_positives)):
        temp = utils.unnorm(torch.Tensor(img)).numpy()
        temp = (temp * 255).astype(np.uint8)
        # utils.show(temp)
        utils.im_save(temp, f'examples/watermarked_positives/positive_{i}.png')

    for i, img in tqdm(enumerate(clean_negatives)):
        temp = utils.unnorm(torch.Tensor(img)).numpy()
        temp = (temp * 255).astype(np.uint8)
        # utils.show(temp)
        utils.im_save(temp, f'examples/watermarked_negatives/negative_{i}.png')

    clean_positives, clean_negatives = test('weights/classifier_quarters_P-match.pth', test_loaders, device)[2]

    for i, img in tqdm(enumerate(clean_positives)):
        temp = utils.unnorm(torch.Tensor(img)).numpy()
        temp = (temp * 255).astype(np.uint8)
        # utils.show(temp)
        utils.im_save(temp, f'examples/watermark_positives/positive_{i}.png')

    for i, img in tqdm(enumerate(clean_negatives)):
        temp = utils.unnorm(torch.Tensor(img)).numpy()
        temp = (temp * 255).astype(np.uint8)
        # utils.show(temp)
        utils.im_save(temp, f'examples/watermark_negatives/negative_{i}.png')

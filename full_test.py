import pickle

import numpy as np
from tqdm import tqdm
import dwn
from train_classify import trim, get_mixed, make_sets, get_test_loaders, train
from test_classify import get_loss
import utils
from torch.utils import data
from torch import nn
import torch


def run(args):
    f = open(args['dataset'], 'rb')
    x, xh, w, y, _ = pickle.load(f)  # Normalized data, ordered as 10 classes x 5000 samples
    f.close()

    f = open(args['test'], 'rb')
    x_test, xh_test, w_test, y_test, _ = pickle.load(f)  # Normalized data, ordered as 10 classes x 1000 samples
    f.close()

    if args['mixed'] is not None:
        classes = trim(x, xh, y)  # Divides classes into clean and marked samples with no overlap
        train_x, train_y = get_mixed(classes, args['mixed'], args['box_mode'], args['mix_size'])  # When training with mixed data

        clean_train, marked_train = make_sets(classes)
        mixed_train = utils.XYDataset(train_x, train_y)
        whole_train = data.ConcatDataset((clean_train, marked_train, mixed_train))
    else:
        whole_train = utils.XYDataset(xh, y)

    train_loader = data.DataLoader(
        whole_train,
        batch_size=args['batch_size'],
        num_workers=0,
        shuffle=True)

    test_loaders = get_test_loaders(x_test, xh_test, w_test, y_test, args['batch_size'])

    model = dwn.Classifier()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters())

    loss_func = nn.CrossEntropyLoss()

    best = 99999.

    for _ in tqdm(range(args['epochs'])):
        loss = train(model, train_loader, test_loaders, optimizer, loss_func, device, use_tqdm=False)
        if loss[args['val_method']] < best:
            best = loss[args['val_method']]
            torch.save(model.state_dict(), args['weight'])

    model.eval()
    test_losses = []
    for test_loader in test_loaders:
        test_losses.append(get_loss(model, test_loader, loss_func, device))

    return test_losses


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repeats = 10

    _ = {
        'dataset': 'datasets/watermark/',  # Dataset given by adversary
        'mixed': 'datasets/watermark/',  # D(mix)
        'mix_size': None,  # Size of D(mix), should change wrt. number and percentage of watermarks
        'box_mode': None,  # Patching method
        'test': 'datasets/watermark/',  # Test set
        'weight': 'weights/',  # Model address
        'epochs': 40,
        'batch_size': 128,
        'val_method': 0,  # 0=clean, 1=marked, 2=watermark
        'name': ''  # pretty print
    }

    test1 = {
        'dataset': 'datasets/watermark/dataset_1.0.pickle',  # Dataset given by adversary
        'mixed': 'datasets/watermark/test_mix.pickle',  # D(mix)
        'mix_size': 450 * 9,  # Size of D(mix), should change wrt. number and percentage of watermarks
        'box_mode': 'random_quarters',  # Patching method
        'test': 'datasets/watermark/dataset_1.0_test.pickle',  # Test set
        'weight': 'weights/classifier_test.pth',  # Model address
        'epochs': 40,
        'batch_size': 128,
        'val_method': 0,  # 0=clean, 1=marked, 2=watermark
        'name': 'Naive'  # pretty print
    }

    watermark = {
        'dataset': 'datasets/watermark/dataset.pickle',
        'mixed': None,
        'mix_size': None,
        'box_mode': None,
        'test': 'datasets/watermark/dataset_test.pickle',
        'weight': 'weights/classifier_watermark.pth',
        'epochs': 40,
        'batch_size': 128,
        'val_method': 1,
        'name': 'Watermarked only'
    }

    debias_quarters_P_match = {
        'dataset': 'datasets/watermark/dataset.pickle',
        'mixed': 'datasets/watermark/quarters_P-match.pickle',
        'mix_size': 4500*9,
        'box_mode': 'random_quarters',
        'test': 'datasets/watermark/dataset_test.pickle',
        'weight': 'weights/classifier_quarters_P-match.pth',
        'epochs': 15,
        'batch_size': 128,
        'val_method': 0,
        'name': 'P-match quarters'
    }

    debias_quarters = {
        'dataset': 'datasets/watermark/dataset.pickle',
        'mixed': 'datasets/watermark/debiased_quarters.pickle',
        'mix_size': 4500,
        'box_mode': 'random_quarters',
        'test': 'datasets/watermark/dataset_test.pickle',
        'weight': 'weights/classifier_quarters.pth',
        'epochs': 30,
        'batch_size': 128,
        'val_method': 0,
        'name': 'Counterexample quarters'
    }

    clean_mixed_marked_validation = {
        'dataset': 'datasets/watermark/dataset.pickle',
        'mixed': 'datasets/watermark/clean_mix.pickle',
        'mix_size': 4500,
        'box_mode': 'id',
        'test': 'datasets/watermark/dataset_test.pickle',
        'weight': 'weights/classifier_clean_mix_marked_validation',
        'epochs': 80,
        'batch_size': 128,
        'val_method': 0,
        'name': 'Clean-mixed'
    }

    debias_halves_P_match = {
        'dataset': 'datasets/watermark/dataset.pickle',
        'mixed': 'datasets/watermark/halves_P-match.pickle',
        'mix_size': 4500*9,
        'box_mode': 'random_halves',
        'test': 'datasets/watermark/dataset_test.pickle',
        'weight': 'weights/classifier_halves_P-match.pth',
        'epochs': 15,
        'batch_size': 128,
        'val_method': 0,
        'name': 'P-match halves'
    }

    debias_quarters_P_match_false = {
        'dataset': 'datasets/watermark/dataset.pickle',
        'mixed': 'datasets/watermark/quarters_P-match.pickle',
        'mix_size': 4500 * 9,
        'box_mode': 'random_quarters',
        'test': 'datasets/watermark/dataset_false_test.pickle',
        'weight': 'weights/classifier_quarters_P-match_false.pth',
        'epochs': 15,
        'batch_size': 128,
        'val_method': 0,
        'name': 'P-match quarters false'
    }

    # For .75 watermarks

    naive75 = {
        'dataset': 'datasets/watermark/dataset_0.75.pickle',  # Dataset given by adversary
        'mixed': None,  # D(mix)
        'mix_size': None,  # Size of D(mix), should change wrt. number and percentage of watermarks
        'box_mode': None,  # Patching method
        'test': 'datasets/watermark/dataset_0.75_test.pickle',  # Test set
        'weight': 'weights/classifier_naive75.pth',  # Model address
        'epochs': 40,
        'batch_size': 128,
        'val_method': 1,  # 0=clean, 1=marked, 2=watermark
        'name': 'Naive'  # pretty print
    }

    quarter75 = {
        'dataset': 'datasets/watermark/dataset_0.75.pickle',  # Dataset given by adversary
        'mixed': 'datasets/watermark/quarters_.75_.25-match.pickle',  # D(mix)
        'mix_size': 10125,  # 4500*.25*9 as integer
        'box_mode': 'random_quarters',  # Patching method
        'test': 'datasets/watermark/dataset_0.75_test.pickle',  # Test set
        'weight': 'weights/classifier_quarter75.pth',  # Model address
        'epochs': 15,
        'batch_size': 128,
        'val_method': 0,  # 0=clean, 1=marked, 2=watermark
        'name': '4500*.25*9'  # pretty print
    }

    half75 = {
        'dataset': 'datasets/watermark/dataset_0.75.pickle',  # Dataset given by adversary
        'mixed': 'datasets/watermark/quarters_.75_.5-match.pickle',  # D(mix)
        'mix_size': 20250,  # 4500*.5*9 as integer
        'box_mode': 'random_quarters',  # Patching method
        'test': 'datasets/watermark/dataset_0.75_test.pickle',  # Test set
        'weight': 'weights/classifier_half75.pth',  # Model address
        'epochs': 15,
        'batch_size': 128,
        'val_method': 0,  # 0=clean, 1=marked, 2=watermark
        'name': '4500*.5*9'  # pretty print
    }

    three_quarters75 = {
        'dataset': 'datasets/watermark/dataset_0.75.pickle',  # Dataset given by adversary
        'mixed': 'datasets/watermark/quarters_.75_.75-match.pickle',  # D(mix)
        'mix_size': 30375,  # 4500*.75*9 as integer
        'box_mode': 'random_quarters',  # Patching method
        'test': 'datasets/watermark/dataset_0.75_test.pickle',  # Test set
        'weight': 'weights/classifier_three_quarters75.pth',  # Model address
        'epochs': 15,
        'batch_size': 128,
        'val_method': 0,  # 0=clean, 1=marked, 2=watermark
        'name': '4500*.75*9'  # pretty print
    }

    full75 = {
        'dataset': 'datasets/watermark/dataset_0.75.pickle',  # Dataset given by adversary
        'mixed': 'datasets/watermark/quarters_.75_1.0-match.pickle',  # D(mix)
        'mix_size': 4500*9,  # Size of D(mix), should change wrt. number and percentage of watermarks
        'box_mode': 'random_quarters',  # Patching method
        'test': 'datasets/watermark/dataset_0.75_test.pickle',  # Test set
        'weight': 'weights/classifier_full75.pth',  # Model address
        'epochs': 15,
        'batch_size': 128,
        'val_method': 0,  # 0=clean, 1=marked, 2=watermark
        'name': '4500*9'  # pretty print
    }

    tenth75 = {
        'dataset': 'datasets/watermark/dataset_0.75.pickle',  # Dataset given by adversary
        'mixed': 'datasets/watermark/quarters_.75_0.1-match.pickle',  # D(mix)
        'mix_size': 450 * 9,  # Size of D(mix), should change wrt. number and percentage of watermarks
        'box_mode': 'random_quarters',  # Patching method
        'test': 'datasets/watermark/dataset_0.75_test.pickle',  # Test set
        'weight': 'weights/classifier_tenth75.pth',  # Model address
        'epochs': 15,
        'batch_size': 128,
        'val_method': 0,  # 0=clean, 1=marked, 2=watermark
        'name': '450*9'  # pretty print
    }

    tenth_fully_watermarked = {
        'dataset': 'datasets/watermark/dataset_1.0.pickle',  # Dataset given by adversary
        'mixed': 'datasets/watermark/quarters_1.0_0.1-match.pickle',  # D(mix)
        'mix_size': 450 * 9,  # Size of D(mix), should change wrt. number and percentage of watermarks
        'box_mode': 'random_quarters',  # Patching method
        'test': 'datasets/watermark/dataset_1.0_test.pickle',  # Test set
        'weight': 'weights/classifier_test.pth',  # Model address
        'epochs': 15,
        'batch_size': 128,
        'val_method': 0,  # 0=clean, 1=marked, 2=watermark
        'name': 'Naive'  # pretty print
    }

    methods = naive75, tenth75, three_quarters75, watermark, tenth_fully_watermarked, debias_quarters_P_match
    metrics = []

    for method in methods:
        metric = []
        for repeat in range(repeats):
            print(f'Repeat {repeat}:')
            metric.append(run(method))
        metrics.append(metric)

    print(metrics)
    f = open('metrics/full_vs_75_metric.pickle', 'wb')
    pickle.dump(metrics, f)
    f.close()

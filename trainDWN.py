import torch
from torch.utils import data
from tqdm import tqdm

import dwn


def train(model, train_loader, test_loader, optimizer, loss_func):
    model.train()
    for i, (x, yx, z, yz) in enumerate(tqdm(train_loader, leave=False)):
        x = x.to(device)
        yx = yx.to(device)
        z = z.to(device)
        yz = yz.to(device)

        ins = (x, z)
        outs = model(x, z)
        ys = (yx, yz)

        loss = loss_func(ins, outs, ys)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return get_loss(model, test_loader, loss_func)


def get_loss(model, train_loader, loss_func):
    loss = 0.
    for i, (x, yx, z, yz) in enumerate(tqdm(train_loader, leave=False)):
        x = x.to(device)
        yx = yx.to(device)
        z = z.to(device)
        yz = yz.to(device)

        ins = (x, z)
        outs = model(x, z)
        ys = (yx, yz)

        loss += float(loss_func(ins, outs, ys).detach().cpu())
    loss /= len(train_loader.dataset)
    return loss


def main():
    print(f'==DWN training with {device}==')

    fashion, cifar = dwn.load_data()

    random_parallel = dwn.RandomParallelDataset(cifar, fashion)

    train_loader = data.DataLoader(
        random_parallel,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True)

    fashion_test, cifar_test = dwn.load_data(train=False)

    random_parallel_test = dwn.RandomParallelDataset(cifar_test, fashion_test)

    test_loader = data.DataLoader(
        random_parallel_test,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True)

    print('Datasets loaded')

    model = dwn.UnderCoverModel()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    loss_func = dwn.DWNLoss()

    print('Ready to train')

    best = 99999.

    for epoch in range(epochs := 100):
        loss = train(model, train_loader, test_loader, optimizer, loss_func)
        if loss < best:
            torch.save(model.state_dict(), weight_path)
        print(f'Epoch: {epoch}, loss: {loss}')


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_path = 'weights/dwn.pth'
    batch_size = 128
    main()

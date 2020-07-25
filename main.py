import os
import argparse
from dataloader import *
from test import test_func
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import torch
import pickle
from model import SAT
import time
import math
from tqdm import tqdm


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    with open(os.path.join(args.path, 'mapping.pkl'), 'rb') as f:
        mapping = pickle.load(f)

    vocab_size = len(mapping)
    sat = SAT(vocab_size=vocab_size)
    sat.to(device)
    loss_func = nn.CrossEntropyLoss(ignore_index=2)
    opt = optim.Adam(sat.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

    if not args.test:
        train_dl = get_loader(mode='train', path=args.path, batch_size=args.batch_size)
        val_dl = get_loader(mode='val', path=args.path, batch_size=args.batch_size)
        best_loss = float('inf')
        for epoch in range(args.epochs):
            pbar = tqdm(train_dl)
            sat.train()
            for xb, yb in pbar:
                loss = sat.train_batch(xb, yb)
                pbar.set_description("| epoch: {:3d} | loss: {:.6f} |".format(epoch + 1, loss))

            sat.eval()
            with torch.no_grad():
                val_losses = []
                for xb, yb in tqdm(val_dl):
                    loss = sat.valid_batch(xb, yb)
                    val_losses.append(loss.detach().cpu().numpy())

                val_loss = np.mean(val_losses)
                print('val loss: {:.6f}'.format(val_loss))
                if best_loss > val_loss:
                    best_loss = val_loss
                    sat.save(file_name='model/best-model.pt', num_epoch=epoch)

    # test
    else:
        print('testing...')
        beam_size = 5
        test_dl = get_loader(mode='test', path=args.path, batch_size=args.batch_size)
        sat.load('./model/best-model.pt')
        with open(os.path.join(args.path, 'itow.pkl'), 'rb') as f:
            itow = pickle.load(f)

        P = []
        for xb, yb in tqdm(test_dl):
            pred = sat.inference(xb, beam_size=5, max_len=36)
            P += pred

        test_func(P, itow)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SAT')
    parser.add_argument(
        '--path',
        type=str,
        default='./dataset')

    parser.add_argument(
        '--epochs',
        type=int,
        default=128)

    parser.add_argument(
        '--batch_size',
        type=int,
        default=40)

    parser.add_argument(
        '--test',
        action='store_true')
    args = parser.parse_args()

    main(args)
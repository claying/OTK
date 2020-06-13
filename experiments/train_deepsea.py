import os
import argparse
import copy

from otk.data_utils import MatDataset
from otk.models_deepsea import SeqAttention
from torch.utils.data import DataLoader, Subset
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau
import torch.optim as optim
import numpy as np
from sklearn.metrics import (
    roc_auc_score, log_loss, accuracy_score,
    precision_recall_curve, average_precision_score
)

from timeit import default_timer as timer

name = 'DeepSEA'
datadir = '../data/{}'.format(name)


def load_args():
    parser = argparse.ArgumentParser(
        description="OT attention for Encode DeepSEA experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--seed', type=int, default=1, metavar='S',
        help='random seed')
    parser.add_argument(
        '--batch-size', type=int, default=128, metavar='M',
        help='input batch size for training')
    parser.add_argument(
        '--epochs', type=int, default=20, metavar='N',
        help='number of epochs to train')
    parser.add_argument(
        "--hidden-size", default=1536, type=int,
        help="hidden size for conv1d and attn layers")
    parser.add_argument(
        "--filter-size", default=16, type=int,
        help="filter size for embedding conv1d")
    parser.add_argument(
        "--weight-decay", type=float, default=1e-06,
        help="regularization parameter for sup CKN")
    parser.add_argument(
        "--eps", default=1.0, type=float, help="eps for OT kernel")
    parser.add_argument(
        "--attn-layers", default=1, type=int, help="number of OT attn layers")
    parser.add_argument(
        '--heads', type=int, default=1, help='number of heads for attention layer')
    parser.add_argument(
        '--out-size', type=int, default=64, help='number of supports for attention layer')
    parser.add_argument(
        '--max-iter', type=int, default=30, help='max iteration for ot kernel')
    parser.add_argument(
        '--hidden-layer', action='store_true', help='use one hidden-layer in classfier')
    parser.add_argument(
        '--position-encoding', default=None, choices=['gaussian', 'hard'],
        help='position encoding type')
    parser.add_argument(
        '--position-sigma', default=0.1, type=float, help='sigma for position encoding')
    parser.add_argument("--lr", type=float, default=0.001, help='initial learning rate')
    parser.add_argument("--report-step", type=int, default=1000,
        help="report stat step during training")
    parser.add_argument(
        "--outdir", default='', type=str, help="output path")
    parser.add_argument(
        "--subset", type=int, default=None, help='train on a subset')
    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()

    args.save_logs = False
    if args.outdir != "":
        args.save_logs = True

    return args

def compute_scores(y_true, y_pred, report_gt_feature_n_positives=10):
    n_features = y_true.shape[1]
    metric_fn = {'auROC': roc_auc_score, 'auPRC': average_precision_score}
    scores = {}
    for key in metric_fn:
        scores[key] = np.empty(n_features)
        scores[key].fill(np.nan)
    for i in range(n_features):
        feat_true = y_true[:, i]
        if np.count_nonzero(feat_true) > report_gt_feature_n_positives:
            for key in metric_fn:
                scores[key][i] = metric_fn[key](feat_true, y_pred[:, i])
    avg_scores = {}
    for key in metric_fn:
        avg_scores[key] = np.nanmean(scores[key])
    return avg_scores, scores

def train_epoch(model, data_loader, criterion, optimizer, val_loader=None,
                report_step=500, use_cuda=False):
    model.train()
    running_loss = 0.0
    running_acc = 0.
    tic = timer()
    for step, (data, label) in enumerate(data_loader):
        size = data.shape[0]
        if use_cuda:
            data = data.cuda()
            label = label.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label) / size
        loss.backward()
        optimizer.step()
        model.normalize_()

        pred = (output.data > 0).float()

        running_loss += loss.item() * size
        running_acc += torch.sum((pred == label.data).float().mean(dim=-1)).item()

        if val_loader is not None and (step + 1) % report_step == 0:
            print('Step {}, Time {:.2f}s'.format(step, timer() - tic))
            eval_epoch(model, val_loader, criterion, use_cuda=use_cuda)
            model.train()
    toc = timer()

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = running_acc / len(data_loader.dataset)
    print('Train Loss: {:.4f} Acc: {:.4f} Time: {:.2f}s'.format(
           epoch_loss, epoch_acc, toc - tic))
    return epoch_loss, epoch_acc

def eval_epoch(model, data_loader, criterion, use_cuda=False):
    model.eval()
    running_loss = 0.0
    y_pred = torch.Tensor(len(data_loader.dataset), model.nclass)
    y_true = torch.Tensor(len(data_loader.dataset), model.nclass)
    index = 0
    tic = timer()
    for data, label in data_loader:
        size = data.shape[0]
        y_true[index: index + size] = label
        if use_cuda:
            data = data.cuda()
            label = label.cuda()

        with torch.no_grad():
            output = model(data)
            loss = criterion(output, label) / size
            y_pred[index: index + size] = output.data.sigmoid().cpu()

        running_loss += loss.item() * size
        index += size
    toc = timer()

    epoch_loss = running_loss / len(data_loader.dataset)
    scores, all_scores = compute_scores(y_true, y_pred)
    print('Val Loss: {:.4f} auROC: {:.4f} auPRC: {:.4f} Time: {:.2f}s'.format(
           epoch_loss, scores['auROC'], scores['auPRC'], toc - tic))
    return epoch_loss, scores, all_scores

def load_checkpoint(args):
    if args.save_logs:
        checkpoint_file = args.outdir + "/checkpoint_last.pt"
        if os.path.isfile(checkpoint_file):
            return torch.load(checkpoint_file)
    return None

def main():
    args = load_args()
    print(args)
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_dset = MatDataset(datadir, 'train')
    val_dset = MatDataset(datadir, 'valid')
    print("training size {}, validation size {}".format(len(train_dset), len(val_dset)))

    loader_args = {}
    if args.use_cuda:
        loader_args = {'num_workers': 1, 'pin_memory': True}

    if args.subset is not None:
        sub_train_dset = Subset(train_dset, range(min(len(train_dset), args.subset)))
        train_loader = DataLoader(
            sub_train_dset, batch_size=args.batch_size, shuffle=True, **loader_args)
    else:
        train_loader = DataLoader(
            train_dset, batch_size=args.batch_size, shuffle=True, **loader_args)

    val_loader = DataLoader(
        val_dset, batch_size=args.batch_size, shuffle=False, **loader_args)

    model = SeqAttention(
        919, args.hidden_size, args.filter_size, args.attn_layers,
        args.eps, args.heads, args.out_size, max_iter=args.max_iter,
        hidden_layer=args.hidden_layer, position_encoding=args.position_encoding,
        position_sigma=args.position_sigma)
    print(model)
    if args.use_cuda:
        model.cuda()

    criterion = nn.BCEWithLogitsLoss(reduction='sum')

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = MultiStepLR(
        optimizer, milestones=[1, 4, 8], gamma=0.5)

    print("Load checkpoint...")
    start_epoch = 0
    best_loss = float('inf')
    checkpoint = load_checkpoint(args)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['weight'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']

    print("Start training...")
    for epoch in range(start_epoch, args.epochs):
        print('Epoch {}/{}'.format(epoch + 1, args.epochs))
        print('-' * 10)
        print("current LR: {}".format(
              optimizer.param_groups[0]['lr']))
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer,
            val_loader=val_loader, report_step=args.report_step, use_cuda=args.use_cuda)
        val_loss, val_scores, _ = eval_epoch(
            model, val_loader, criterion, use_cuda=args.use_cuda)

        if val_loss < best_loss:
            best_loss = val_loss
            best_val_scores = val_scores
            best_epoch = epoch
            if args.save_logs:
                torch.save({
                    'epoch': best_epoch,
                    'best_loss': best_loss,
                    'optimizer': optimizer.state_dict(),
                    'weight': model.state_dict(),
                    },
                    args.outdir + "/checkpoint_best.pt"
                )

        if args.save_logs:
            import shutil
            checkpoint_file = args.outdir + "/checkpoint{}.pt".format(epoch)
            checkpoint_last = args.outdir + "/checkpoint_last.pt"
            torch.save({
                'epoch': epoch,
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
                'weight': model.state_dict(),
                },
                checkpoint_file
            )
            shutil.copyfile(checkpoint_file, checkpoint_last)

        if isinstance(lr_scheduler, ReduceLROnPlateau):
            lr_scheduler.step(val_loss)
        else:
            lr_scheduler.step()

    print("Testing...")
    if args.save_logs:
        checkpoint = torch.load(args.outdir + "/checkpoint_best.pt")
        model.load_state_dict(checkpoint['weight'])

    test_dset = MatDataset(datadir, 'test')
    test_loader = DataLoader(
        test_dset, batch_size=args.batch_size, shuffle=False, **loader_args)
    test_loss, test_avg_score, test_scores = eval_epoch(
        model, test_loader, criterion, use_cuda=args.use_cuda)


    if args.save_logs:
        import pandas as pd
        test_avg_score = pd.DataFrame.from_dict(test_avg_score, orient='index')
        test_avg_score.to_csv(args.outdir + '/metric.csv',
                  header=['value'], index_label='metric')
        for key in test_scores:
            np.savetxt(args.outdir + '/{}.txt'.format(key), test_scores[key])
    return


if __name__ == "__main__":
    main()

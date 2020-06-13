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

def compute_scores(y_true, y_pred, report_gt_feature_n_positives=0):
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

def eval_epoch(model, data_loader, criterion, return_pred=False, use_cuda=False):
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
    scores, all_scores = compute_scores(y_true.numpy(), y_pred.numpy())
    print('Val Loss: {:.4f} auROC: {:.4f} auPRC: {:.4f} Time: {:.2f}s'.format(
           epoch_loss, scores['auROC'], scores['auPRC'], toc - tic))
    if return_pred:
        return epoch_loss, scores, all_scores, y_true, y_pred
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

    if args.save_logs:
        print("Loading model...")
        checkpoint = torch.load(args.outdir + "/checkpoint_best.pt")
        model.load_state_dict(checkpoint['weight'])

    print("Validating")
    eval_epoch(model, val_loader, criterion, use_cuda=args.use_cuda)

    print("Testing...")
    test_dset = MatDataset(datadir, 'test')
    test_loader = DataLoader(
        test_dset, batch_size=args.batch_size, shuffle=False, **loader_args)
    test_loss, test_avg_score, test_scores, y_true, y_pred = eval_epoch(
        model, test_loader, criterion, return_pred=True, use_cuda=args.use_cuda)


    if args.save_logs:
        import pandas as pd
        test_avg_score = pd.DataFrame.from_dict(test_avg_score, orient='index')
        test_avg_score.to_csv(args.outdir + '/metric.csv',
                  header=['value'], index_label='metric')
        np.savez_compressed(args.outdir + '/ypreds', ytrue=y_true.numpy(), ypred=y_pred.numpy())
        for key in test_scores:
            np.savetxt(args.outdir + '/{}.txt'.format(key), test_scores[key])
    return


if __name__ == "__main__":
    main()

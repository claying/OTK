import os
import argparse

from loaders import load_data, load_masks
from otk.models import SeqAttention
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau
import torch.optim as optim
import numpy as np
import copy

from timeit import default_timer as timer


def load_args():
    parser = argparse.ArgumentParser(
        description="unsup OT kernel for SST2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', choices=['sst-2_bert_mask',
                                              'sst-2_proto'],
                        default='sst-2_bert_mask')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed')
    parser.add_argument(
        '--batch-size', type=int, default=128,
        help='input batch size for training')
    parser.add_argument(
        "--n-filters", default=[128], nargs='+', type=int,
        help="number of filters for each layer")
    parser.add_argument(
        "--len-motifs", default=[1], nargs='+', type=int,
        help="filter size for each layer")
    parser.add_argument(
        "--subsamplings", default=[1], nargs='+', type=int,
        help="subsampling for each layer")
    parser.add_argument(
        "--kernel-params", default=[0.6], nargs='+', type=float,
        help="sigma for Gaussian kernel at each layer")
    parser.add_argument(
        "--sampling-patches", default=300000, type=int,
        help="number of sampled patches")
    parser.add_argument(
        "--weight-decay", type=float, default=0.0001,
        help="weight decay for classifier")
    parser.add_argument(
        '--eps', type=float, default=0.5, help='eps for Sinkhorn')
    parser.add_argument(
        '--heads', type=int, default=1,
        help='number of heads for attention layer')
    parser.add_argument(
        '--out-size', type=int, default=30,
        help='number of supports for attention layer')
    parser.add_argument(
        '--max-iter', type=int, default=100,
        help='max iteration for ot kernel')
    parser.add_argument(
        '--wb', action='store_true',
        help='use Wasserstein barycenter instead of kmeans')
    parser.add_argument(
        '--baseline', type=str, default='ours', choices=['ours'])
    parser.add_argument(
        "--outdir", default="results/", type=str, help="output path")
    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()
    # check shape
    assert len(args.n_filters) == len(args.len_motifs) == len(
        args.subsamplings) == len(args.kernel_params), "numbers mismatched"
    args.n_layers = len(args.n_filters)

    args.save_logs = False
    if args.outdir != "":
        args.save_logs = True
        outdir = args.outdir + args.dataset
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        outdir = outdir + "/unsup"
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        outdir = outdir + f"/{args.baseline}"
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        outdir = outdir+'/ckn_{}_{}_{}_{}'.format(
            args.n_filters, args.len_motifs, args.subsamplings,
            args.kernel_params)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        outdir = outdir + '/{}_{}_{}_{}_{}_{}'.format(
            args.max_iter, args.eps, args.out_size, args.heads,
            args.weight_decay, args.wb)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        args.outdir = outdir

    return args


def accuracy(output, target):
    assert(len(output) == len(target))
    acc = torch.sum(torch.argmax(output, dim=1) == target).float()
    return acc / len(target)


def eval_epoch(model, data_loader, criterion, use_cuda=False):
    model.eval()
    running_loss = 0.0
    running_acc = 0.
    for data, label in data_loader:
        size = data.shape[0]
        if use_cuda:
            data = data.cuda()
            label = label.cuda()

        with torch.no_grad():
            output = model(data)
            loss = criterion(output, label)
            pred = output.data.argmax(dim=1)

        running_loss += loss.item() * size
        running_acc += torch.sum(pred == label.data).item()

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = running_acc / len(data_loader.dataset)
    return epoch_loss, epoch_acc


def eval_epoch_list(model, data_loaders, criterion, use_cuda=False):
    epoch_loss = []
    epoch_acc = []
    tic = timer()
    for _, v_loader in data_loaders:
        e_loss, e_acc = eval_epoch(
            model, v_loader, criterion, use_cuda=use_cuda)
        epoch_loss.append(e_loss)
        epoch_acc.append(e_acc)
    toc = timer()
    epoch_loss = np.mean(epoch_loss)
    epoch_acc = np.mean(epoch_acc)
    print('Val Loss: {:.4f} Acc: {:.4f} Time: {:.2f}s'.format(
           epoch_loss, epoch_acc, toc - tic))
    return epoch_loss, epoch_acc


def preprocess(X):
    X -= X.mean(dim=-1, keepdim=True)
    X /= X.norm(dim=-1, keepdim=True)
    return X


def main():
    args = load_args()
    print(args)
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    X_train, y_train, X_val, y_val, _ = load_data(dataset=args.dataset)
    mask_train, mask_val, _ = load_masks(args.dataset)

    X_train *= mask_train
    X_val *= mask_val

    y_train = torch.from_numpy(y_train)
    y_val = torch.from_numpy(y_val)

    nb_train = int(0.8 * X_train.shape[0])

    train_dset = TensorDataset(X_train[:nb_train].permute(0, 2, 1),
                               y_train[:nb_train])
    val_dset = TensorDataset(X_train[nb_train:].permute(0, 2, 1),
                             y_train[nb_train:])

    loader_args = {}
    if args.use_cuda:
        loader_args = {'num_workers': 1, 'pin_memory': True}

    train_loader = DataLoader(
        train_dset, batch_size=args.batch_size, shuffle=False, **loader_args)
    val_loader = DataLoader(
        val_dset, batch_size=args.batch_size, shuffle=False, **loader_args)

    model = SeqAttention(
        768, 2, args.n_filters, args.len_motifs, args.subsamplings,
        kernel_args=args.kernel_params, alpha=args.weight_decay,
        eps=args.eps, heads=args.heads, out_size=args.out_size,
        max_iter=args.max_iter, fit_bias=False)
    print(model)
    print(len(train_dset))

    print("Initializing...")
    tic = timer()
    if args.use_cuda:
        model.cuda()
    n_samples = 3000
    if args.n_filters[-1] > 256:
        n_samples //= args.n_filters[-1] // 256
    model.unsup_train(train_loader, args.sampling_patches, n_samples=n_samples,
                      wb=args.wb, use_cuda=args.use_cuda)
    toc = timer()
    print("Finished feature learning, elapsed time: {:.2f}s".format(toc - tic))

    print("Encoding...")
    Xtr, ytr = model.predict(train_loader, only_repr=True,
                             use_cuda=args.use_cuda)
    preprocess(Xtr)
    print(Xtr.shape)

    Xval = []
    yval = []

    X, y = model.predict(val_loader, only_repr=True, use_cuda=args.use_cuda)
    preprocess(X)
    Xval.append(X)
    yval.append(y)

    search_grid = 2. ** np.arange(1, 15)
    search_grid = 1. / search_grid
    best_score = -np.inf
    clf = model.classifier
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    if Xtr.shape[-1] > 20000:
        optimizer = torch.optim.Adam(clf.parameters(), lr=0.01)
        epochs = 800
    else:
        optimizer = torch.optim.LBFGS(
                clf.parameters(), lr=1.0, max_eval=10, history_size=10,
                tolerance_grad=1e-05, tolerance_change=1e-05)
        epochs = 100
    torch.cuda.empty_cache()
    print("Start crossing validation")
    for alpha in search_grid:
        tic = timer()
        clf.fit(Xtr, ytr, criterion, reg=alpha, epochs=epochs,
                optimizer=optimizer, use_cuda=args.use_cuda)
        toc = timer()
        scores = []
        for X, y in zip(Xval, yval):
            if args.use_cuda:
                X = X.cuda()
            score = clf.score(X, y)
            scores.append(score)
        score = np.mean(scores)
        print("CV alpha={}, acc={:.2f}, ts={:.2f}s".format(alpha, score * 100.,
              toc - tic))
        if score > best_score:
            best_score = score
            best_alpha = alpha
            best_weight = copy.deepcopy(clf.state_dict())

    clf.load_state_dict(best_weight)

    print("Finished, elapsed time: {:.2f}s".format(toc - tic))

    test_dset = TensorDataset(X_val.permute(0, 2, 1), y_val)
    test_loader = DataLoader(
        test_dset, batch_size=args.batch_size, shuffle=False)
    Xte, y_true = model.predict(test_loader, only_repr=True,
                                use_cuda=args.use_cuda)
    preprocess(Xte)
    if args.use_cuda:
        Xte = Xte.cuda()
    with torch.no_grad():
        y_pred = clf(Xte).cpu()

    scores = accuracy(y_pred, y_true)
    print(scores)

    if args.save_logs:
        print('Saving logs...')
        data = {
            # 'title': title,
            'score': scores,
            'best_param': best_alpha,
            'val_score': best_score,
            'args': args
            }
        np.save(os.path.join(args.outdir, f"seed_{args.seed}_results.npy"),
                data)
        # torch.save(
        #     {'args': args,
        #      'state_dict': model.state_dict()},
        #     args.outdir + '/model.pkl')
    return


if __name__ == "__main__":
    main()

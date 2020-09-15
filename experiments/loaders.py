import numpy as np
import torch
import os


def load_cifar_features(path, one_vs_all=None):
    data = np.load(path)
    X, y = data['X'], data['y']
    if one_vs_all is not None:
        y_ = np.zeros(len(y))
        for i in range(len(X)):
            if y[i] == one_vs_all:
                y_[i] = 1
    X = torch.from_numpy(X)
    X = X.view((X.shape[0], X.shape[1] * X.shape[2], X.shape[3]))
    X_val = torch.empty(0)
    y_val = torch.empty(0)
    X_test = torch.empty(0)
    return X, y, X_val, y_val, X_test


def load_sst2_features(dataset):
    if dataset == 'sst-2_bert':
        data_train = np.load('sst-2/sst_train_bert-base-uncased.npz')
        data_val = np.load('../data/sst-2/sst_val_bert-base-uncased.npz')
        data_test = np.load('../data/sst-2/sst_test_bert-base-uncased.npz')
    elif dataset == 'sst-2_bert_finetuned':
        data_train = np.load('../data/sst-2/sst_train_bert-base-uncased_finetuned.npz')
        data_val = np.load('../data/sst-2/sst_val_bert-base-uncased_finetuned.npz')
        data_test = np.load('../data/sst-2/sst_test_bert-base-uncased_finetuned.npz')
    elif dataset == 'sst-2_bert_mask':
        data_train = np.load('../data/sst-2/sst_train_bert-base-uncased_mask.npz')
        data_val = np.load('../data/sst-2/sst_val_bert-base-uncased_mask.npz')
        data_test = np.load('../data/sst-2/sst_test_bert-base-uncased_mask.npz')
    elif dataset == 'sst-2_proto':
        data_train = np.load('../data/sst-2/sst_val_bert-base-uncased_mask.npz')
        data_val = np.load('../data/sst-2/sst_val_bert-base-uncased_mask.npz')
        data_test = np.load('../data/sst-2/sst_test_bert-base-uncased_mask.npz')
    elif dataset == 'sst-2_bert_wordemb_mask':
        data_train = np.load('../data/sst-2/sst_train_wordemb_bert-base-uncased_mask.npz')
        data_val = np.load('../data/sst-2/sst_val_wordemb_bert-base-uncased_mask.npz')
        data_test = np.load('../data/sst-2/sst_test_wordemb_bert-base-uncased_mask.npz')
    elif dataset == 'sst-2_bert_large':
        data_train = np.load('../data/sst-2/sst_train_bert-large-uncased.npz')
        data_val = np.load('../data/sst-2/sst_val_bert-large-uncased.npz')
        data_test = np.load('../data/sst-2/sst_test_bert-large-uncased.npz')
    elif dataset == 'sst-2_bert_66':
        data_train = np.load('../data/sst-2/sst_train_66_bert-base-uncased.npz')
        data_val = np.load('../data/sst-2/sst_val_66_bert-base-uncased.npz')
        data_test = np.load('../data/sst-2/sst_test_66_bert-base-uncased.npz')
    elif dataset == 'sst-2_bert_66_mask':
        data_train = np.load('../data/sst-2/sst_train_66_bert-base-uncased_mask.npz')
        data_val = np.load('../data/sst-2/sst_val_66_bert-base-uncased_mask.npz')
        data_test = np.load('../data/sst-2/sst_test_66_bert-base-uncased_mask.npz')
    elif dataset == 'sst-2_bert_66_finetuned':
        data_train = np.load('../data/sst-2/sst_train_66_bert-base-uncased_finetuned.npz')
        data_val = np.load('../data/sst-2/sst_val_66_bert-base-uncased_finetuned.npz')
        data_test = np.load('../data/sst-2/sst_test_66_bert-base-uncased_finetuned.npz')
    elif dataset == 'sst-2_roberta-base':
        data_train = np.load('../data/sst-2/sst_train_roberta-base.npz')
        data_val = np.load('../data/sst-2/sst_val_roberta-base.npz')
        data_test = np.load('../data/sst-2/sst_test_roberta-base.npz')
    # the above operation is fast.
    X_train, y_train = data_train['X'], data_train['y'] # this is where things are slow.
    X_val, y_val = data_val['X'], data_val['y']
    X_test = data_test['X']
    X_train = torch.from_numpy(X_train)
    X_val = torch.from_numpy(X_val)
    X_test = torch.from_numpy(X_test)
    return X_train, y_train, X_val, y_val, X_test


def load_data(dataset):
    if dataset == 'cifar_5k_256':
        X_train, y_train, X_val, y_val, X_test = load_cifar_features(
            '../data/cifar-10/ckn64_256.npz')
    elif dataset == 'cifar_5k_256_16x16':
        X_train, y_train, X_val, y_val, X_test = load_cifar_features(
            '../data/cifar-10/ckn64_256_16x16.npz')
    elif dataset == 'cifar_5k_8k':
        X_train, y_train, X_val, y_val, X_test = load_cifar_features(
            '../data/cifar-10/ckn512_8192.npz')
    elif 'sst-2' in dataset:
        X_train, y_train, X_val, y_val, X_test = load_sst2_features(dataset)
    print('Train', X_train.shape, y_train.shape)
    print('Val', X_val.shape, y_val.shape)
    print('Test', X_test.shape)
    return X_train, y_train, X_val, y_val, X_test


def load_masks(dataset):
    if '66' in dataset:
        mask_length = 66
    else:
        mask_length = 30
    X_val_mask = torch.from_numpy(np.load(
        '../data/sst-2/sst_val_bert_{}_mask.npz'.format(mask_length))['masks']
        ).unsqueeze(dim=-1)
    if 'proto' in dataset:
        X_tr_mask = X_val_mask
    else:
        X_tr_mask = torch.from_numpy(np.load(
            '../data/sst-2/sst_train_bert_{}_mask.npz'.format(mask_length)
            )['masks']).unsqueeze(dim=-1)
    X_test_mask = torch.from_numpy(np.load(
        '../data/sst-2/sst_test_bert_{}_mask.npz'.format(mask_length))['masks']
        ).unsqueeze(dim=-1)
    return X_tr_mask, X_val_mask, X_test_mask

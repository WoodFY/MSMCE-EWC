import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

import os
import random
import argparse
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight

from cumulative_trainer import train_cumulative
from losses.ewc import EWC
from callbacks.early_stopping import EarlyStopping


os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'


def set_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    # Set Python random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set PyTorch random seed
    torch.manual_seed(seed)

    # Ensure reproducibility in multi-GPU training
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_bin_dataset_path(exp_args):
    pass


def prepare_dataset():
    pass


def exp(exp_args, save_dir, label_mapping, device, use_multi_gpu=False):
    print(f"{exp_args['model_name']} {exp_args['dataset']}_dataset num_classes {exp_args['num_classes']} in_channels: {exp_args['in_channels']} embedding_channels: {exp_args['embedding_channels']} embedding_dim: {exp_args['embedding_dim']}")
    exp_dir_name = (f"{exp_args['model_name']}_{exp_args['dataset']}_dataset_num_classes_{exp_args['num_classes']}_in_channels_{exp_args['in_channels']}"
                    f"embedding_channels_{exp_args['embedding_channels']}_embedding_dim_{exp_args['embedding_dim']}_batch_size_{exp_args['batch_size']}")
    exp_dir = os.path.join(
        save_dir,
        exp_dir_name
    )

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    tasks = prepare_dataset(
        tasks_info=exp_args.tasks_info,
        label_mapping = label_mapping
    )

    if len(tasks) >= 2:
        exp_name = f'train_{tasks[0].dataset_name}_cumulative_' + '_'.join([task.dataset_name for task in tasks[1:]])
    else:
        raise ValueError('Not enough datasets for cumulative learning.')

    exp_args['exp_dir'] = exp_dir
    exp_args['exp_name'] = exp_name

    model = None
    ewc_regularizer = EWC(args=exp_args)

    if use_multi_gpu and torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs for training.')
        device = torch.device('cuda:0')
        model = model.to(device)
        model = nn.DataParallel(model)
    else:
        model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters, lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-32)
    if exp_args['is_early_stopping']:
        early_stopping = EarlyStopping(patience=7, verbose=True, delta=0.01)
    else:
        early_stopping = None

    train_cumulative(
        args=exp_args,
        model=model,
        tasks=tasks,
        criterion=criterion,
        optimizer=optimizer,
        ewc_regularizer=None,
        scheduler=scheduler,
        early_stopping=early_stopping,
        fisher_estimate_sample_size=1024,
        consolidate=True,
        is_early_stopping=exp_args['is_early_stopping'],
        is_metrics_visualization=True
    )


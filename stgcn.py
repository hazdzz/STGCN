import logging
import os
import argparse
import math
import random
import tqdm
import numpy as np
import pandas as pd
from sklearn import preprocessing

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils

from script import dataloader, utility, earlystopping
from model import models

#import nni

def set_env(seed):
    # Set available CUDA devices
    # This option is crucial for multiple GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_parameters():
    parser = argparse.ArgumentParser(description='STGCN')
    parser.add_argument('--enable_cuda', type=bool, default='True', help='enable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=42, help='set the random seed for stabilize experiment results')
    parser.add_argument('--dataset', type=str, default='metr-la', choices=['metr-la', 'pems-bay', 'pemsd7-m'])
    parser.add_argument('--n_his', type=int, default=12)
    parser.add_argument('--n_pred', type=int, default=3, help='the number of time interval for predcition, default as 3')
    parser.add_argument('--time_intvl', type=int, default=5)
    parser.add_argument('--Kt', type=int, default=3)
    parser.add_argument('--stblock_num', type=int, default=2)
    parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu'])
    parser.add_argument('--Ks', type=int, default=3, choices=[3, 2])
    parser.add_argument('--graph_conv_type', type=str, default='cheb_graph_conv', choices=['cheb_graph_conv', 'graph_conv'])
    parser.add_argument('--gso_type', type=str, default='sym_norm_lap', choices=['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj'])
    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay_rate', type=float, default=0.0005, help='weight decay (L2 penalty)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10000, help='epochs, default as 10000')
    parser.add_argument('--opt', type=str, default='adam', help='optimizer, default as adam')
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--patience', type=int, default=30, help='early stopping patience')
    args = parser.parse_args()
    print('Training configs: {}'.format(args))

    # For stable experiment results
    SEED = args.seed
    set_env(SEED)

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        # Set available CUDA devices
        # This option is crucial for multiple GPUs
        # 'cuda' ≡ 'cuda:0'
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if args.dataset == 'metr-la' or args.dataset == 'pems-bay' or args.dataset == 'pemsd7-m':
        dataset = args.dataset
    else:
        raise ValueError(f'ERROR: {args.dataset} is not existed.')
    
    n_his = args.n_his
    n_pred = args.n_pred
    time_intvl = args.time_intvl
    Kt = args.Kt
    stblock_num = args.stblock_num
    Ko = n_his - (Kt - 1) * 2 * stblock_num
    if args.act_func == 'glu' or args.act_func == 'gtu':
        act_func = args.act_func
    else:
        raise NotImplementedError(f'ERROR: {args.act_func} is not defined.')
    Ks = args.Ks
    if args.graph_conv_type == 'cheb_graph_conv':
        if args.gso_type == 'sym_norm_lap' or args.gso_type == 'rw_norm_lap':
            graph_conv_type = args.graph_conv_type
            gso_type = args.gso_type
        else:
            raise ValueError(f'ERROR: {args.gso_type} is not matched with {args.graph_conv_type}')
    elif args.graph_conv_type == 'graph_conv':
        if args.gso_type == 'sym_renorm_adj' or args.gso_type == 'rw_renorm_adj':
            graph_conv_type = args.graph_conv_type
            gso_type = args.gso_type
        else:
            raise ValueError(f'ERROR: {args.gso_type} is not matched with {args.graph_conv_type}')
    else:
        raise NotImplementedError(f'ERROR: {args.graph_conv_type} is not defined.')

    # blocks: settings of channel size in st_conv_blocks and output layer,
    # using the bottleneck design in st_conv_blocks
    blocks = []
    blocks.append([1])
    for l in range(stblock_num):
        blocks.append([64, 16, 64])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([1])
    
    n_pred = args.n_pred
    time_pred = n_pred * time_intvl
    time_pred_str = str(time_pred) + '_mins'

    enable_bias = args.enable_bias
    droprate = args.droprate
    lr = args.lr
    weight_decay_rate = args.weight_decay_rate
    batch_size = args.batch_size
    epochs = args.epochs
    opt = args.opt
    step_size = args.step_size
    gamma = args.gamma
    patience = args.patience

    model_save_dir = os.path.join('./model/save', dataset)
    os.makedirs(name=model_save_dir, exist_ok=True)
    model_save_path =  'stgcn_' + gso_type + '_' + time_pred_str + '.pth'
    model_save_path = os.path.join(model_save_dir, model_save_path)
    
    return device, dataset, n_his, n_pred, time_intvl, Kt, stblock_num, act_func, Ks, graph_conv_type, gso_type, blocks, n_pred, time_pred, enable_bias, droprate, lr, weight_decay_rate, batch_size, epochs, opt, step_size, gamma, patience, model_save_path

def data_preparate(dataset, gso_type, graph_conv_type, n_his, n_pred, batch_size, device):    
    adj, n_vertex = dataloader.load_adj(dataset)
    gso = utility.calc_gso(adj, gso_type)
    if graph_conv_type == 'cheb_graph_conv':
        gso = utility.calc_chebynet_gso(gso)
    gso = gso.toarray()
    gso = gso.astype(dtype=np.float32)
    gso = torch.from_numpy(gso).to(device)

    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset)
    data_col = pd.read_csv(os.path.join(dataset_path, 'vel.csv')).shape[0]
    # recommended dataset split rate as train: val: test = 60: 20: 20, 70: 15: 15 or 80: 10: 10
    # using dataset split rate as train: val: test = 70: 15: 15
    val_and_test_rate = 0.15

    len_val = int(math.floor(data_col * val_and_test_rate))
    len_test = int(math.floor(data_col * val_and_test_rate))
    len_train = int(data_col - len_val - len_test)

    train, val, test = dataloader.load_data(dataset, len_train, len_val)
    zscore = preprocessing.StandardScaler()
    train = zscore.fit_transform(train)
    val = zscore.transform(val)
    test = zscore.transform(test)

    x_train, y_train = dataloader.data_transform(train, n_his, n_pred, device)
    x_val, y_val = dataloader.data_transform(val, n_his, n_pred, device)
    x_test, y_test = dataloader.data_transform(test, n_his, n_pred, device)

    train_data = utils.data.TensorDataset(x_train, y_train)
    train_iter = utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    val_data = utils.data.TensorDataset(x_val, y_val)
    val_iter = utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
    test_data = utils.data.TensorDataset(x_test, y_test)
    test_iter = utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    return gso, n_vertex, zscore, train_iter, val_iter, test_iter

def prepare_model(patience, model_save_path, Kt, Ks, blocks, n_his, n_vertex, act_func, graph_conv_type, gso, enable_bias, droprate, device, opt, lr, weight_decay_rate, step_size, gamma):
    loss = nn.MSELoss()
    early_stopping = earlystopping.EarlyStopping(patience=patience, path=model_save_path, verbose=True)

    if graph_conv_type == 'cheb_graph_conv':
        model = models.STGCNChebGraphConv(Kt, Ks, blocks, n_his, n_vertex, act_func, graph_conv_type, gso, enable_bias, droprate).to(device)
    else:
        model = models.STGCNGraphConv(Kt, Ks, blocks, n_his, n_vertex, act_func, graph_conv_type, gso, enable_bias, droprate).to(device)

    if opt == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay_rate)
    elif opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay_rate, amsgrad=False)
    elif opt == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay_rate, amsgrad=False)
    else:
        raise NotImplementedError(f'ERROR: The optimizer {opt} is not implemented.')

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    return loss, early_stopping, model, optimizer, scheduler

def train(loss, epochs, optimizer, scheduler, early_stopping, model, train_iter, val_iter):
    min_val_loss = np.inf
    for epoch in range(epochs):
        l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        model.train()
        for x, y in tqdm.tqdm(train_iter):
            y_pred = model(x).view(len(x), -1)  # [batch_size, num_nodes]
            l = loss(y_pred, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            scheduler.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        val_loss = val(model, val_iter)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
        early_stopping(val_loss, model)
        # GPU memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print('Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB'.\
            format(epoch+1, optimizer.param_groups[0]['lr'], l_sum / n, val_loss, gpu_mem_alloc))

        if early_stopping.early_stop:
            print("Early stopping.")
            break
    print('\nTraining finished.\n')

def val(model, val_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in val_iter:
            y_pred = model(x).view(len(x), -1)
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n
    
def test(model_save_path, zscore, loss, model, test_iter, dataset):
    model.load_state_dict(torch.load(model_save_path))
    test_MSE = utility.evaluate_model(model, loss, test_iter)
    test_MAE, test_RMSE, test_WMAPE = utility.evaluate_metric(model, test_iter, zscore)
    print(f'Dataset {dataset:s} | Test loss {test_MSE:.6f} | MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f} | WMAPE {test_WMAPE:.8f}')

if __name__ == "__main__":
    # Logging
    #logger = logging.getLogger('stgcn')
    #logging.basicConfig(filename='stgcn.log', level=logging.INFO)
    logging.basicConfig(level=logging.INFO)

    device, dataset, n_his, n_pred, time_intvl, Kt, stblock_num, act_func, Ks, graph_conv_type, gso_type, blocks, n_pred, time_pred, enable_bias, droprate, lr, weight_decay_rate, batch_size, epochs, opt, step_size, gamma, patience, model_save_path = get_parameters()
    gso, n_vertex, zscore, train_iter, val_iter, test_iter = data_preparate(dataset, gso_type, graph_conv_type, n_his, n_pred, batch_size, device)
    loss, early_stopping, model, optimizer, scheduler = prepare_model(patience, model_save_path, Kt, Ks, blocks, n_his, n_vertex, act_func, graph_conv_type, gso, enable_bias, droprate, device, opt, lr, weight_decay_rate, step_size, gamma)
    train(loss, epochs, optimizer, scheduler, early_stopping, model, train_iter, val_iter)
    test(model_save_path, zscore, loss, model, test_iter, dataset)
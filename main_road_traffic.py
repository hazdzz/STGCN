import logging
import os
import argparse
import configparser
import tqdm
import numpy as np
import pandas as pd
from sklearn import preprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils as utils

from torchsummary import summary

from script import dataloader, utility, earlystopping
from model import models

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='STGCN for road traffic prediction')
parser.add_argument('--enable_cuda', type=bool, default='True',
                    help='enable CUDA, default as True')
parser.add_argument('--time_intvl', type=int, default=5,
                    help='time interval of sampling (mins), default as 5 mins')
parser.add_argument('--n_pred', type=int, default=9, 
                    help='the number of time interval for predcition, default as 9 (literally means 45 mins)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size, defualt as 32')
parser.add_argument('--epochs', type=int, default=500,
                    help='epochs, default as 500')
parser.add_argument('--config_path', type=str, default='./config/chebconv_sym_glu.ini',
                    help='the path of config file, chebconv_sym_glu.ini for STGCN(ChebConv, Ks=3), \
                    and gcnconv_sym_glu.ini for STGCN(GCNConv)')
parser.add_argument('--dropout_rate', type=float, default=0.2,
                    help='dropout rate, default as 0.2')
parser.add_argument('--opt', type=str, default='AdamW',
                    help='optimizer, default as AdamW')
parser.add_argument('--data_path', type=str, default='./data/road_traffic/PeMS-M/V_228.csv',
                    help='the path of road traffic data')
parser.add_argument('--wam_path', type=str, default='./data/road_traffic/PeMS-M/A_228.csv',
                    help='the path of weighted adjacency matrix')
args = parser.parse_args()
print('Training configs: {}'.format(args))

# Running in Nvidia GPU (CUDA) or CPU
if args.enable_cuda and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

config = configparser.ConfigParser()
config_path = args.config_path
config.read(config_path, encoding="utf-8")

def ConfigSectionMap(section):
    dict1 = {}
    options = config.options(section)
    for option in options:
        try:
            dict1[option] = config.get(section, option)
            if dict1[option] == -1:
                logging.debug("skip: %s" % option)
        except:
            print("exception on %s!" % option)
            dict1[option] = None
    return dict1

Kt = int(ConfigSectionMap('casualconv')['kt'])
if (Kt != 2) and (Kt != 3):
    raise ValueError(f'ERROR: Kt must be 2 or 3, "{Kt}" is unacceptable, unless you rewrite the code.')
else:
    Kt = Kt
graph_conv_type = ConfigSectionMap('graphconv')['graph_conv_type']
if (graph_conv_type != "chebconv") and (graph_conv_type != "gcnconv"):
    raise NotImplementedError(f'ERROR: "{graph_conv_type}" is not implemented.')
else:
    graph_conv_type = graph_conv_type
Ks = int(ConfigSectionMap('graphconv')['ks'])
if (graph_conv_type == 'gcnconv') and (Ks != 2):
    Ks = 2
mat_type = ConfigSectionMap('graphconv')['mat_type']

# blocks: settings of channel size in st_conv_blocks and output layer,
# using the bottleneck design in st_conv_blocks
blocks = [[1, 64, 16, 64], [64, 64, 16, 64], [64, 128, 128, 128]]
if (args.time_intvl % 2 == 0) or (args.time_intvl % 3 == 0) or (args.time_intvl % 5 == 0):
    time_intvl = args.time_intvl
else:
    raise ValueError(f'ERROR: time_intvl must be n times longer than 2, 3 or 5, "{args.time_intvl}" is unacceptable.')
day_slot = int(24 * 60 / time_intvl)
n_pred = args.n_pred
n_his = int(12)

time_pred = n_pred * time_intvl
time_pred_str = '_'+str(time_pred)+'_mins'
checkpoint_path = ConfigSectionMap('graphconv')['checkpoint_path']
checkpoint_path = checkpoint_path + time_pred_str + '.pth'
model_save_path = ConfigSectionMap('graphconv')['model_save_path']
model_save_path = model_save_path + time_pred_str + '.pth'

wam_path = args.wam_path
adj_mat = dataloader.load_weighted_adjacency_matrix(wam_path)

n_train, n_val, n_test = 34, 5, 5
len_train, len_val, len_test = n_train * day_slot, n_val * day_slot, n_test * day_slot
data_path = args.data_path
n_vertex_v = pd.read_csv(data_path, header=None).shape[1]
n_vertex_a = pd.read_csv(wam_path, header=None).shape[1]
if n_vertex_v != n_vertex_a:
    raise ValueError(f'ERROR: number of vertices in dataset is not equal to number of vertices in weighted adjacency matrix.')
else:
    n_vertex = n_vertex_v

dropout_rate = args.dropout_rate

if graph_conv_type == "chebconv":
    mat = utility.calculate_laplacian_metrix(adj_mat, mat_type)
    graph_conv_filter_list = utility.calculate_chebconv_graph_filter(mat, Ks)
    chebconv_filter_list = torch.from_numpy(graph_conv_filter_list).float().to(device)
    stgcn_chebconv = models.STGCN_ChebConv(Kt, Ks, blocks, n_his, n_vertex, graph_conv_type, chebconv_filter_list, dropout_rate).to(device)
    if (mat_type != "wid_sym_normd_lap_mat") and (mat_type != "wid_rw_normd_lap_mat"):
        raise ValueError(f'ERROR: "{args.mat_type}" is wrong.')
elif graph_conv_type == "gcnconv":
    mat = utility.calculate_laplacian_metrix(adj_mat, mat_type)
    gcnconv_filter = torch.from_numpy(mat).float().to(device)
    stgcn_gcnconv = models.STGCN_GCNConv(Kt, Ks, blocks, n_his, n_vertex, graph_conv_type, gcnconv_filter, dropout_rate).to(device)
    if (mat_type != "hat_sym_normd_lap_mat") and (mat_type != "hat_rw_normd_lap_mat"):
        raise ValueError(f'ERROR: "{args.mat_type}" is wrong.')

train, val, test = dataloader.load_data(data_path, len_train, len_val)
zscore = preprocessing.StandardScaler()
train = zscore.fit_transform(train)
val = zscore.transform(val)
test = zscore.transform(test)

x_train, y_train = dataloader.data_transform(train, n_his, n_pred, day_slot, device)
x_val, y_val = dataloader.data_transform(val, n_his, n_pred, day_slot, device)
x_test, y_test = dataloader.data_transform(test, n_his, n_pred, day_slot, device)

bs = args.batch_size
train_data = utils.data.TensorDataset(x_train, y_train)
train_iter = utils.data.DataLoader(dataset=train_data, batch_size=bs, shuffle=False)
val_data = utils.data.TensorDataset(x_val, y_val)
val_iter = utils.data.DataLoader(dataset=val_data, batch_size=bs, shuffle=False)
test_data = utils.data.TensorDataset(x_test, y_test)
test_iter = utils.data.DataLoader(dataset=test_data, batch_size=bs, shuffle=False)

loss = nn.MSELoss()
epochs = args.epochs
learning_rate = 7.5e-4
early_stopping = earlystopping.EarlyStopping(patience=30, path=checkpoint_path, verbose=True)

if graph_conv_type == "chebconv":
    model = stgcn_chebconv
    model_stats = summary(stgcn_chebconv, (1, n_his, n_vertex))
elif graph_conv_type == "gcnconv":
    model = stgcn_gcnconv
    model_stats = summary(stgcn_gcnconv, (1, n_his, n_vertex))

if args.opt == "RMSProp":
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
elif args.opt == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
elif args.opt == "AdamW":
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
else:
    raise ValueError(f'ERROR: optimizer "{args.opt}" is undefined.')

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.999)

def val():
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in val_iter:
            y_pred = model(x).view(len(x), -1)
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n

def train():
    valid_losses = []
    min_val_loss = np.inf
    for epoch in range(1, epochs + 1):
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
        val_loss = val()
        valid_losses.append(val_loss)
        valid_loss = np.average(valid_losses)
        # GPU memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
        print('Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.2f} MiB'.\
            format(epoch, optimizer.param_groups[0]['lr'], l_sum / n, val_loss, gpu_mem_alloc))

        # clear lists to track next epoch
        valid_losses = []
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping.")
            break
    print('\nTraining finished.\n')
    
def test():
    if graph_conv_type == "chebconv":
        best_model = stgcn_chebconv
    elif graph_conv_type == "gcnconv":
        best_model = stgcn_gcnconv
    best_model.load_state_dict(torch.load(model_save_path))
    test_MSE = utility.evaluate_model(best_model, loss, test_iter)
    print('Test loss {:.6f}'.format(test_MSE))
    test_MAE, test_MAPE, test_RMSE = utility.evaluate_metric(best_model, test_iter, zscore)
    print('MAE {:.6f} | MAPE {:.8f} | RMSE {:.6f}'.format(test_MAE, test_MAPE, test_RMSE))

if __name__ == "__main__":
    train()
    test()

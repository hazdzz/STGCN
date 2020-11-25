import os
import argparse
import math
import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchsummary import summary

from script import data_loader, utils, pytorchtools
from model import stgcn

parser = argparse.ArgumentParser(description='STGCN for PM2.5 prediction')
parser.add_argument('--enable_cuda', type=bool, default='True',
                    help='enable CUDA, default as True')
parser.add_argument('--window_size', type=int, default=20,
                    help='sampling period (mins)')
parser.add_argument('--t_pred', type=int, default=60,
                    help='the period of prediction')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size, defualt as 32')
parser.add_argument('--epochs', type=int, default=100,
                    help='epochs, default as 100')
parser.add_argument('--gnn', type=str, default='ChebNet',
                    help='the type of GNN, default as ChebNet, you could choose GCN as alternative')
parser.add_argument('--Kt', type=int, default=3,
                    help='the kernel size of causal convolution, default as 3')
parser.add_argument('--Ks', type=int, default=3,
                    help='the kernel size of graph convolution with Chebshev Polynomials approximation(ChebNet), defulat as 3, for first-order approximation(GCN), Ks = 1')
parser.add_argument('--opt', type=str, default='Adam',
                    help='optimizer, default as Adam')
parser.add_argument('--data_path', type=str, default='./data/pm25/non_st_cal/pm25_ksh.csv',
                    help='the path of PM2.5 data')
parser.add_argument('--wam_path', type=str, default='./data/pm25/non_st_cal/wam_ksh.csv',
                    help='the path of weighted adjacency matrix')
parser.add_argument('--model_stgcn_chebnet_save_path', type=str, default='./model/save/stgcn_chebnet_pm25.pt',
                    help='the save path of model STGCN(ChebNet) for PM2.5')
parser.add_argument('--model_stgcn_gcn_save_path', type=str, default='./model/save/stgcn_gcn_pm25.pt',
                    help='the save path of model STGCN(GCN) for PM2.5')
args = parser.parse_args()
print('Training configs: {}'.format(args))

# Running in Nvidia GPU (CUDA) or CPU
if args.enable_cuda and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Kt is the kernel size of casual convolution, default as 3
# Ks is the kernel size of ChebNet(Chebyshev Polynomials approximation), default as 3
# The kernel size of GCN(First Order Chebyshev Polynomials approximation) as 1, i.e., Ks = 1
Kt, Ks = args.Kt, args.Ks
# blocks: settings of channel size in st_conv_blocks / bottleneck design
blocks = [[1, 32, 64], [64, 32, 128]]
if (args.window_size % 2 == 0) or (args.window_size % 3 == 0) or (args.window_size % 5 == 0):
    window_size = args.window_size
else:
    raise ValueError(f'ERROR: window_size must be n times longer than 2, 3 or 5, "{args.window_size}" is unacceptable.')
day_slot = int(24 * 60 / window_size)
n_pred = math.floor(args.t_pred / window_size)
n_his = max(n_pred, (len(blocks) * 2 * (Kt - 1) + 2))

wam_path = args.wam_path
W = data_loader.load_weighted_adjacency_matrix(wam_path)
widetilde_L = utils.scaled_laplacian(W)
cheb_poly = torch.from_numpy(np.array(utils.cheb_poly_approx(widetilde_L, Ks))).float().to(device)
first_order_cheb_poly = torch.from_numpy(np.array(utils.first_order_cheb_poly_approx(W))).float().to(device)

n_train, n_val, n_test = 46, 23, 23
len_train, len_val, len_test = n_train * day_slot, n_val * day_slot, n_test * day_slot
data_path = args.data_path
n_vertex_v = pd.read_csv(data_path, header=None).shape[1]
n_vertex_w = pd.read_csv(wam_path, header=None).shape[1]
if n_vertex_v != n_vertex_w:
    raise ValueError(f'ERROR: number of vertices in dataset is not equal to number of vertices in weighted adjacency matrix.')
else:
    n_vertex = n_vertex_v

train, val, test = data_loader.load_data(data_path, len_train, len_val)
scaler = StandardScaler()
train = scaler.fit_transform(train)
val = scaler.transform(val)
test = scaler.transform(test)

x_train, y_train = data_loader.data_transform(train, n_his, n_pred, day_slot, device)
x_val, y_val = data_loader.data_transform(val, n_his, n_pred, day_slot, device)
x_test, y_test = data_loader.data_transform(test, n_his, n_pred, day_slot, device)

bs = args.batch_size
train_data = torch.utils.data.TensorDataset(x_train, y_train)
train_iter = torch.utils.data.DataLoader(dataset=train_data, batch_size=bs, shuffle = False)
val_data = torch.utils.data.TensorDataset(x_val, y_val)
val_iter = torch.utils.data.DataLoader(dataset=val_data, batch_size=bs, shuffle = False)
test_data = torch.utils.data.TensorDataset(x_test, y_test)
test_iter = torch.utils.data.DataLoader(dataset=test_data, batch_size=bs, shuffle = False)

drop_prob = 0.1
loss = nn.MSELoss()
epochs = args.epochs
if (args.gnn != "ChebNet") or (args.gnn != "GCN"):
    gnn = args.gnn
else:
    raise ValueError(f'ERROR: "{args.gnn}" is undefined.')
model_stgcn_chebnet = stgcn.STGCN_ChebNet(Kt, Ks, blocks, n_his, n_vertex, gnn, cheb_poly, drop_prob).to(device)
model_stgcn_chebnet_save_path = args.model_stgcn_chebnet_save_path
model_stgcn_gcn = stgcn.STGCN_GCN(Kt, 1, blocks, n_his, n_vertex, gnn, first_order_cheb_poly, drop_prob=0).to(device)
model_stgcn_gcn_save_path = args.model_stgcn_gcn_save_path
if gnn == "ChebNet":
    model = model_stgcn_chebnet
    model_save_path = model_stgcn_chebnet_save_path
    model_stats = summary(model_stgcn_chebnet, (1, n_his, n_vertex))
    early_stopping = pytorchtools.EarlyStopping(patience=20, path="./model/checkpoint/cp_stgcn_chebnet_pm25.pt",verbose=True)
    learning_rate = 5e-4
elif gnn == "GCN":
    model = model_stgcn_gcn
    model_save_path = model_stgcn_gcn_save_path
    model_stats = summary(model_stgcn_gcn, (1, n_his, n_vertex))
    early_stopping = pytorchtools.EarlyStopping(patience=20, path="./model/checkpoint/cp_stgcn_gcn_pm25.pt",verbose=True)
    learning_rate = 16e-5
if args.opt == "RMSProp":
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
elif args.opt == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
else:
    raise ValueError(f'ERROR: optimizer "{args.opt}" is undefined.')
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.999)

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
        print('Epoch: {:03d} | Lr: {:.10f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.2f} MiB'.format(epoch, optimizer.param_groups[0]['lr'], l_sum / n, val_loss, gpu_mem_alloc))

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
    if gnn == "ChebNet":
        best_model = stgcn.STGCN_ChebNet(Kt, Ks, blocks, n_his, n_vertex, gnn, cheb_poly, drop_prob).to(device)
    elif gnn == "GCN":
        best_model = stgcn.STGCN_GCN(Kt, 1, blocks, n_his, n_vertex, gnn, first_order_cheb_poly, drop_prob=0).to(device)
    best_model.load_state_dict(torch.load(model_save_path))
    test_MSE = utils.evaluate_model(best_model, loss, test_iter)
    print('Test loss {:.6f}'.format(test_MSE))
    test_MAE, test_MAPE, test_RMSE = utils.evaluate_metric(best_model, test_iter, scaler)
    print('MAE {:.6f} | RMSE {:.6f} | MAPE {:.6f}'.format(test_MAE, test_RMSE, test_MAPE))

if __name__ == "__main__":
    train()
    test()

import numpy as np
from scipy.linalg import eigvalsh
from scipy.linalg import fractional_matrix_power

import torch

def calculate_laplacian_matrix(adj_mat, mat_type):
    n_vertex = adj_mat.shape[0]
    id_mat = np.identity(n_vertex)

    # D_row
    deg_mat_row = np.diag(np.sum(adj_mat, axis=1))
    # D_com
    #deg_mat_col = np.diag(np.sum(adj_mat, axis=0))

    # D = D_row as default
    deg_mat = deg_mat_row

    # wid_A = A + I
    wid_adj_mat = adj_mat + id_mat
    # wid_D = D + I
    wid_deg_mat = deg_mat + id_mat

    # Combinatorial Laplacian
    # L_com = D - A
    com_lap_mat = deg_mat - adj_mat

    if mat_type == 'id_mat':
        return id_mat
    elif mat_type == 'com_lap_mat':
        return com_lap_mat

    if (mat_type == 'sym_normd_lap_mat') or (mat_type == 'wid_sym_normd_lap_mat') or (mat_type == 'hat_sym_normd_lap_mat'):
        deg_mat_inv_sqrt = fractional_matrix_power(deg_mat, -0.5)
        wid_deg_mat_inv_sqrt = fractional_matrix_power(wid_deg_mat, -0.5)

        # Symmetric normalized Laplacian
        # For SpectraConv
        # L_sym = D^{-0.5} * L_com * D^{-0.5} = I - D^{-0.5} * A * D^{-0.5}
        sym_normd_lap_mat = np.matmul(np.matmul(deg_mat_inv_sqrt, com_lap_mat), deg_mat_inv_sqrt)

        # For ChebConv
        # wid_L_sym = 2 * L_sym / lambda_max_sym - I
        ev_max_sym = max(eigvalsh(sym_normd_lap_mat))
        wid_sym_normd_lap_mat = 2 * sym_normd_lap_mat / ev_max_sym - id_mat

        # For GCNConv
        # hat_L_sym = wid_D^{-0.5} * wid_A * wid_D^{-0.5}
        hat_sym_normd_lap_mat = np.matmul(np.matmul(wid_deg_mat_inv_sqrt, wid_adj_mat), wid_deg_mat_inv_sqrt)

        if mat_type == 'sym_normd_lap_mat':
            return sym_normd_lap_mat
        elif mat_type == 'wid_sym_normd_lap_mat':
            return wid_sym_normd_lap_mat
        elif mat_type == 'hat_sym_normd_lap_mat':
            return hat_sym_normd_lap_mat

    elif (mat_type == 'rw_normd_lap_mat') or (mat_type == 'wid_rw_normd_lap_mat') or (mat_type == 'hat_rw_normd_lap_mat'):

        deg_mat_inv = np.linalg.inv(deg_mat)
        wid_deg_mat_inv = np.linalg.inv(wid_deg_mat)

        # Random Walk normalized Laplacian
        # For SpectraConv
        # L_rw = D^{-1} * L_com = I - D^{-1} * A
        rw_normd_lap_mat = np.matmul(deg_mat_inv, com_lap_mat)

        # For ChebConv
        # wid_L_rw = 2 * L_rw / lambda_max_rw - I
        ev_max_rw = max(eigvalsh(rw_normd_lap_mat))
        wid_rw_normd_lap_mat = 2 * rw_normd_lap_mat / ev_max_rw - id_mat

        # For GCNConv
        # hat_L_rw = wid_D^{-1} * wid_A
        hat_rw_normd_lap_mat = np.matmul(wid_deg_mat_inv, wid_adj_mat)

        if mat_type == 'rw_normd_lap_mat':
            return rw_normd_lap_mat
        elif mat_type == 'wid_rw_normd_lap_mat':
            return wid_rw_normd_lap_mat
        elif mat_type == 'hat_rw_normd_lap_mat':
            return hat_rw_normd_lap_mat

def evaluate_model(model, loss, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x).view(len(x), -1)
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        mse = l_sum / n
        
        return mse

def evaluate_metric(model, data_iter, scaler):
    model.eval()
    with torch.no_grad():
        mae, sum_y, mape, mse = [], [], [], []
        for x, y in data_iter:
            y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
            y_pred = scaler.inverse_transform(model(x).view(len(x), -1).cpu().numpy()).reshape(-1)
            d = np.abs(y - y_pred)
            mae += d.tolist()
            sum_y += y.tolist()
            mape += (d / y).tolist()
            mse += (d ** 2).tolist()
        MAE = np.array(mae).mean()
        MAPE = np.array(mape).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        WMAPE = np.sum(np.array(mae)) / np.sum(np.array(sum_y))

        #return MAE, MAPE, RMSE
        return MAE, RMSE, WMAPE

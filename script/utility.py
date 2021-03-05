import numpy as np

from scipy.linalg import fractional_matrix_power
from scipy.sparse.linalg import eigs

import torch

def calculate_laplacian_matrix(adj_mat, mat_type):
    n_vertex = adj_mat.shape[0]

    # row sum
    deg_mat_row = np.asmatrix(np.diag(np.sum(adj_mat, axis=1)))
    # column sum
    #deg_mat_col = np.asmatrix(np.diag(np.sum(adj_mat, axis=0)))
    deg_mat = deg_mat_row
    
    adj_mat = np.asmatrix(adj_mat)
    id_mat = np.asmatrix(np.identity(n_vertex))

    # Combinatorial
    com_lap_mat = deg_mat - adj_mat

    # For SpectraConv
    # To [0, 1]
    sym_normd_lap_mat = np.matmul(np.matmul(fractional_matrix_power(deg_mat, -0.5), com_lap_mat), fractional_matrix_power(deg_mat, -0.5))

    # For ChebConv
    # From [0, 1] to [-1, 1]
    lambda_max_sym = eigs(sym_normd_lap_mat, k=1, which='LR')[0][0].real
    wid_sym_normd_lap_mat = 2 * sym_normd_lap_mat / lambda_max_sym - id_mat

    # For GCNConv
    wid_deg_mat = deg_mat + id_mat
    wid_adj_mat = adj_mat + id_mat
    hat_sym_normd_lap_mat = np.matmul(np.matmul(fractional_matrix_power(wid_deg_mat, -0.5), wid_adj_mat), fractional_matrix_power(wid_deg_mat, -0.5))

    # Random Walk
    rw_lap_mat = np.matmul(np.linalg.matrix_power(deg_mat, -1), adj_mat)

    # For SpectraConv
    # To [0, 1]
    rw_normd_lap_mat = id_mat - rw_lap_mat

    # For ChebConv
    # From [0, 1] to [-1, 1]
    lambda_max_rw = eigs(rw_lap_mat, k=1, which='LR')[0][0].real
    wid_rw_normd_lap_mat = 2 * rw_normd_lap_mat / lambda_max_rw - id_mat

    # For GCNConv
    wid_deg_mat = deg_mat + id_mat
    wid_adj_mat = adj_mat + id_mat
    hat_rw_normd_lap_mat = np.matmul(np.linalg.matrix_power(wid_deg_mat, -1), wid_adj_mat)

    if mat_type == 'id_mat':
        return id_mat
    elif mat_type == 'com_lap_mat':
        return com_lap_mat
    elif mat_type == 'sym_normd_lap_mat':
        return sym_normd_lap_mat
    elif mat_type == 'wid_sym_normd_lap_mat':
        return wid_sym_normd_lap_mat
    elif mat_type == 'hat_sym_normd_lap_mat':
        return hat_sym_normd_lap_mat
    elif mat_type == 'rw_lap_mat':
        return rw_lap_mat
    elif mat_type == 'rw_normd_lap_mat':
        return rw_normd_lap_mat
    elif mat_type == 'wid_rw_normd_lap_mat':
        return wid_rw_normd_lap_mat
    elif mat_type == 'hat_rw_normd_lap_mat':
        return hat_rw_normd_lap_mat
    else:
        raise ValueError(f'ERROR: "{mat_type}" is unknown.')

def calculate_chebconv_graph_matrix_list(lap_mat, Ks):
    # The Chebyshev polynomials are recursively defined as 
    # T_k(x) = 2 * x * T_{k - 1}(x) - T_{k - 2}(x)
    # T_0(x) = 1
    # T_1(x) = x

    n_vertex = lap_mat.shape[0]
    id_mat = np.identity(n_vertex)

    # K_cp is the order of Chebyshev polynomials
    # K_cp + 1 = Ks
    # Because K_cp starts from 0, and Ks starts from 1 
    K_cp = Ks - 1

    if K_cp == 0:
        # T_0(x) = 1
        return id_mat
    elif K_cp == 1:
        # T_1(x) = x
        chebyshev_polynomials_laplacian_matrix_list = [id_mat]
        chebyshev_polynomials_laplacian_matrix_list.append(lap_mat)
        return np.concatenate(chebyshev_polynomials_laplacian_matrix_list, axis=-1)
    elif K_cp >= 2:
        chebyshev_polynomials_laplacian_matrix_list = [id_mat, lap_mat]

        # T_k(x) = 2 * x * T_{k - 1}(x) - T_{k - 2}(x)
        for k in range(2, Ks):
            chebyshev_polynomials_laplacian_matrix_list.append(2 * lap_mat * chebyshev_polynomials_laplacian_matrix_list[k - 1] - chebyshev_polynomials_laplacian_matrix_list[k - 2])

        return np.concatenate(chebyshev_polynomials_laplacian_matrix_list, axis=-1)
    else:
        raise ValueError(f'ERROR: the graph convolution kernel size Ks must be greater than 0, but received "{Ks}".')

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

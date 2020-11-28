import numpy as np
import scipy
import torch

def scaled_laplacian(W):
    # W is a weighted adjacency matrix
    # D is a diagonal degree matrix
    # \widetilde L = 2 * L / lambda_max - I_n

    n_vertex = W.shape[0]
    D = np.sum(W, axis=1)
    L = np.diag(D) - W

    for i in range(n_vertex):
        for j in range(n_vertex):
            if (D[i] > 0) and (D[j] > 0):
                L[i, j] = L[i, j] / np.sqrt(D[i] * D[j])
    
    lambda_max = scipy.sparse.linalg.eigs(L, k = 1, which = 'LR')[0][0].real
    #lambda_max = np.linalg.eigvals(L).max().real
 
    widetilde_L = 2 * L / lambda_max - np.identity(n_vertex)

    return widetilde_L

def gc_cpa_kernel(widetilde_L, Ks):
    # The Chebyshev Polynomials are recursively defined as 
    # T_k(x) = 2 * x * T_{k - 1}(x) - T_{k - 2}(x)
    # T_0(x) = 1
    # T_1(x) = x

    n_vertex = widetilde_L.shape[0]

    K_cp = Ks - 1

    if K_cp == 0:
        # T_0(x) = 1
        return np.identity(n_vertex)
    elif K_cp == 1:
        # T_0(x) = 1
        # T_1(x) = x
        return [np.identity(n_vertex), widetilde_L]
    elif K_cp >= 2:
        chebyshev_polynomials = [np.identity(n_vertex), widetilde_L]

        # T_k(x) = 2 * x * T_{k - 1}(x) - T_{k - 2}(x)
        for k in range(2, Ks):
            chebyshev_polynomials.append(2 * widetilde_L * chebyshev_polynomials[k - 1] - chebyshev_polynomials[k - 2])

        return np.concatenate(chebyshev_polynomials, axis=-1)
    else:
        raise ValueError(f'ERROR: the order k of Chebyshev Polynomial cannot negative, but received "{K_cp}".')

def gc_lwl_kernel(W):
    n_vertex = W.shape[0]
    widetilde_W = W + np.identity(n_vertex)
    row_sum = np.sum(widetilde_W, axis=1)
    widetilde_D_inv_sqrt = np.mat(np.diag(np.power(row_sum, -0.5)))

    # Renormalized trick
    # \widetildeD^{-1/2} * \widetildeW * \widetildeD^{-1/2}
    return widetilde_D_inv_sqrt * widetilde_W * widetilde_D_inv_sqrt

def evaluate_model(model, loss, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x).view(len(x), -1)
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n

def evaluate_metric(model, data_iter, scaler):
    model.eval()
    with torch.no_grad():
        mae, mse, mape = [], [], []
        for x, y in data_iter:
            y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
            y_pred = scaler.inverse_transform(model(x).view(len(x), -1).cpu().numpy()).reshape(-1)
            d = np.abs(y - y_pred)
            mae += d.tolist()
            mse += (d ** 2).tolist()
            mape += (d / y).tolist()
        MAE = np.array(mae).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        MAPE = np.array(mape).mean()

        return MAE, RMSE, MAPE
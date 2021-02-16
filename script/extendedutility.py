import numpy as np
from scipy.linalg import fractional_matrix_power
from scipy.sparse.linalg import eigs

def calculate_laplacian_matrix(adj_mat, mat_type, enable_trade_off_lambda, trade_off_lambda):
    n_vertex = adj_mat.shape[0]

    # row sum
    deg_mat_row = np.asmatrix(np.diag(np.sum(adj_mat, axis=1)))

    # column sum
    deg_mat_col = np.asmatrix(np.diag(np.sum(adj_mat, axis=0)))

    deg_mat = deg_mat_row

    adj_mat = np.asmatrix(adj_mat)
    id_mat = np.asmatrix(np.identity(n_vertex))

    if enable_trade_off_lambda == True:
        trade_off_lambda = trade_off_lambda
    else:
        trade_off_lambda = 1

    # Symmetric
    com_lap_mat = deg_mat - adj_mat

    # For SpectraConv
    # To (0, 1)
    sym_normd_lap_mat = id_mat - np.matmul(np.matmul(fractional_matrix_power(deg_mat, -0.5), adj_mat), fractional_matrix_power(deg_mat, -0.5))

    # For ChebConv
    # From (0, 1) to (-1, 1)
    lambda_max_sym = eigs(sym_normd_lap_mat, k=1, which='LR')[0][0].real
    wid_sym_normd_lap_mat = 2 * sym_normd_lap_mat / lambda_max_sym - id_mat

    # For GCNConv or GCNIIConv
    wid_deg_mat = deg_mat + trade_off_lambda * id_mat
    wid_adj_mat = adj_mat + trade_off_lambda * id_mat
    hat_sym_normd_lap_mat = np.matmul(np.matmul(fractional_matrix_power(wid_deg_mat, -0.5), wid_adj_mat), fractional_matrix_power(wid_deg_mat, -0.5))

    # Random Walk
    rw_lap_mat = np.matmul(np.linalg.matrix_power(deg_mat, -1), adj_mat)

    # For SpectraConv
    # To (0, 1)
    rw_normd_lap_mat = id_mat - rw_lap_mat

    # For ChebConv
    # From (0, 1) to (-1, 1)
    lambda_max_rw = eigs(rw_lap_mat, k=1, which='LR')[0][0].real
    wid_rw_normd_lap_mat = 2 * rw_normd_lap_mat / lambda_max_rw - id_mat

    # For GCNConv or GCNIIConv
    wid_deg_mat = deg_mat + trade_off_lambda * id_mat
    wid_adj_mat = adj_mat + trade_off_lambda * id_mat
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

def calculate_rwr_adj_mat(lap_mat, alpha):
    n_vertex = lap_mat.shape[0]
    mat = np.identity(n_vertex) - np.multiply(lap_mat, (1 - alpha))
    rwr_adj_mat = np.multiply(np.power(mat, -1), alpha)

    return rwr_adj_mat

def calculate_ppr_adj_mat(lap_mat, alpha):
    return calculate_rwr_adj_mat(lap_mat, alpha)

def calculate_hk_lap_mat(lap_mat, t):
    hk_adj_mat = np.exp(np.multiply(lap_mat, -t))

    return hk_adj_mat

def calculate_laplacian_matrix_in_ppr_or_hk(adj_mat, mat_type):
    n_vertex = adj_mat.shape[0]
    
    # row sum
    deg_mat_row = np.asmatrix(np.diag(np.sum(adj_mat, axis=1)))
    
    # column sum
    deg_mat_col = np.asmatrix(np.diag(np.sum(adj_mat, axis=0)))
    
    deg_mat = deg_mat_row

    adj_mat = np.asmatrix(adj_mat)

    # Symmetric
    sym_lap_mat = np.matmul(np.matmul(fractional_matrix_power(deg_mat, -0.5), adj_mat), fractional_matrix_power(deg_mat, -0.5))

    # Random Walk
    rw_lap_mat = np.matmul(fractional_matrix_power(deg_mat, -1), adj_mat)

    if mat_type == 'sym_lap_mat':
        return sym_lap_mat
    elif mat_type == 'rw_lap_mat':
        return rw_lap_mat
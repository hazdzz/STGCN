import torch
import torch.nn as nn

from model import layers

class STGCN_ChebConv(nn.Module):
    # STGCN(ChebConv) contains 'TGTND TGTND TNTF' structure
    # ChebConv is the graph convolution from ChebyNet.
    # Using the Chebyshev polynomials of the first kind to
    # approximate graph convolution kernel from Spectral CNN.
    # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/cheb_conv.html#ChebConv
        
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normalization
    # T: Temporal Convolution Layer (Sigmoid)
    # F: Fully-Connected Layer

    def __init__(self, Kt, Ks, begin_channel, blocks, end_channel, T, n_vertex, gated_act_func, graph_conv_type, chebconv_matrix_list, drop_rate):
        super(STGCN_ChebConv, self).__init__()
        self.st_block1 = layers.STConvBlock(Kt, Ks, n_vertex, begin_channel, blocks[0], gated_act_func, graph_conv_type, chebconv_matrix_list, drop_rate)
        self.st_block2 = layers.STConvBlock(Kt, Ks, n_vertex, blocks[0][-1] , blocks[1], gated_act_func, graph_conv_type, chebconv_matrix_list, drop_rate)
        Ko = T - (len(blocks) - 1) * 2 * (Kt - 1)
        if Ko > 1:
            self.output = layers.OutputBlock(Ko, blocks[-2][-1], blocks[-1], end_channel, n_vertex, gated_act_func, drop_rate)
        else:
            raise ValueError(f'ERROR: kernel size Ko must be greater than 1, but received "{Ko}".')

    def forward(self, x):
        x_stb1 = self.st_block1(x)
        x_stb2 = self.st_block2(x_stb1)
        x_out = self.output(x_stb2)
        return x_out

class STGCN_GCNConv(nn.Module):
    # STGCN(GCNConv) contains 'TGTND TGTND TNTF' structure
    # GCNConv is the graph convolution from GCN.
    # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gcn_conv.html#GCNConv
        
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (GCNConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (GCNConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normalization
    # T: Temporal Convolution Layer (Sigmoid)
    # F: Fully-Connected Layer

    def __init__(self, Kt, Ks, begin_channel, blocks, end_channel, T, n_vertex, gated_act_func, graph_conv_type, gcnconv_matrix, drop_rate):
        super(STGCN_GCNConv, self).__init__()
        self.st_block1 = layers.STConvBlock(Kt, Ks, n_vertex, begin_channel, blocks[0], gated_act_func, graph_conv_type, gcnconv_matrix, drop_rate)
        self.st_block2 = layers.STConvBlock(Kt, Ks, n_vertex, blocks[0][-1] , blocks[1], gated_act_func, graph_conv_type, gcnconv_matrix, drop_rate)
        Ko = T - (len(blocks) - 1) * 2 * (Kt - 1)
        if Ko > 1:
            self.output = layers.OutputBlock(Ko, blocks[-2][-1], blocks[-1], end_channel, n_vertex, gated_act_func, drop_rate)
        else:
            raise ValueError(f'ERROR: kernel size Ko must be greater than 1, but received "{Ko}".')

    def forward(self, x):
        x_stb1 = self.st_block1(x)
        x_stb2 = self.st_block2(x_stb1)
        x_out = self.output(x_stb2)
        return x_out
import torch
import torch.nn as nn

from model import layers

class STGCN_ChebConv(nn.Module):
    # STGCN(ChebConv) contains 'TSTN TSTN TNTF' structure
    # ChebConv is the graph convolution from ChebyNet.
    # Using the Chebyshev polynomials of the first kind to
    # approximate graph convolution kernel from Spectral CNN.
    # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/cheb_conv.html#ChebConv
        
    # T: Temporal Convolution Layer (GLU)
    # S: Spitial Convolution Layer (ChebConv)
    # T: Temporal Convolution Layer (ReLU)
    # N: Layer Normolization

    # T: Temporal Convolution Layer (GLU)
    # S: Spitial Convolution Layer (ChebConv)
    # T: Temporal Convolution Layer (ReLU)
    # N: Layer Normolization

    # T: Temporal Convolution Layer (GLU)
    # N: Layer Normalization
    # T: Temporal Convolution Layer (Sigmoid)
    # F: Fully-Connected Layer

    def __init__(self, Kt, Ks, blocks, T, n_vertex, graph_conv_type, chebconv_filter_list, dropout_rate):
        super(STGCN_ChebConv, self).__init__()
        self.st_block1 = layers.STConvBlock(Kt, Ks, n_vertex, blocks[0], graph_conv_type, chebconv_filter_list, dropout_rate)
        self.st_block2 = layers.STConvBlock(Kt, Ks, n_vertex, blocks[1], graph_conv_type, chebconv_filter_list, dropout_rate)
        Ko = T - (len(blocks) - 1) * 2 * (Kt - 1)
        if Ko > 1:
            self.output = layers.OutputBlock(blocks[-1], Ko, n_vertex, dropout_rate)
        else:
            raise ValueError(f'ERROR: kernel size Ko must be greater than 1, but received "{Ko}".')

    def forward(self, x):
        x_stb1 = self.st_block1(x)
        x_stb2 = self.st_block2(x_stb1)
        x_out = self.output(x_stb2)
        return x_out

class STGCN_GCNConv(nn.Module):
    # STGCN(GCNConv) contains 'TSTN TSTN TNTF' structure
    # GCNConv is the graph convolution from GCN.
    # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gcn_conv.html#GCNConv
        
    # T: Temporal Convolution Layer (GLU)
    # S: Spitial Convolution Layer (GCNConv)
    # T: Temporal Convolution Layer (ReLU)
    # N: Layer Normolization

    # T: Temporal Convolution Layer (GLU)
    # S: Spitial Convolution Layer (GCNConv)
    # T: Temporal Convolution Layer (ReLU)
    # N: Layer Normolization

    # T: Temporal Convolution Layer (GLU)
    # N: Layer Normalization
    # T: Temporal Convolution Layer (Sigmoid)
    # F: Fully-Connected Layer

    def __init__(self, Kt, Ks, blocks, T, n_vertex, graph_conv_type, gcnconv_filter, dropout_rate):
        super(STGCN_GCNConv, self).__init__()
        self.st_block1 = layers.STConvBlock(Kt, Ks, n_vertex, blocks[0], graph_conv_type, gcnconv_filter, dropout_rate)
        self.st_block2 = layers.STConvBlock(Kt, Ks, n_vertex, blocks[1], graph_conv_type, gcnconv_filter, dropout_rate)
        Ko = T - (len(blocks) - 1) * 2 * (Kt - 1)
        if Ko > 1:
            self.output = layers.OutputBlock(blocks[-1], Ko, n_vertex, dropout_rate)
        else:
            raise ValueError(f'ERROR: kernel size Ko must be greater than 1, but received "{Ko}".')

    def forward(self, x):
        x_stb1 = self.st_block1(x)
        x_stb2 = self.st_block2(x_stb1)
        x_out = self.output(x_stb2)
        return x_out

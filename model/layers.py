import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.conv1x1 = nn.Conv2d(self.c_in, self.c_out, (1, 1))

    def forward(self, x):
        if self.c_in > self.c_out:
            x_self = self.conv1x1(x)
        elif self.c_in < self.c_out:
            batch_size, c, timestep, n_vertex = x.shape
            x_self = torch.cat([x, torch.zeros([batch_size, self.c_out - c, timestep, n_vertex]).to(x)], dim = 1)
        else:
            x_self = x
        return x_self

class TemporalConvLayer(nn.Module):

    # Temporal Convolution Layer (GLU)
    #
    #        |-------------------------------| * residual connection *
    #        |                               |
    #        |    |--->--- casual conv ----- + -------|       
    # -------|----|                                   ⊙ ------>
    #             |--->--- casual conv --- sigmoid ---|                               
    #

    #param x: tensor, [batch_size, c_in, timestep, n_vertex]

    def __init__(self, Kt, c_in, c_out, act_func):
        super(TemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.act_func = act_func
        self.align = Align(c_in, c_out)
        if self.act_func == "GLU":
            self.conv = nn.Conv2d(c_in, 2 * c_out, (Kt, 1), 1)
        else:
            self.conv = nn.Conv2d(c_in, c_out, (Kt, 1), 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):   
        x_input = self.align(x)[:, :, self.Kt - 1:, :]
        x_conv = self.conv(x)

        if self.act_func == "GLU":
            # Temporal Convolution Layer (GLU)
            P = x_conv[:, : self.c_out, :, :]
            Q = x_conv[:, -self.c_out:, :, :]
            P_with_rc = P + x_input
            # (P + x_input) ⊙ Sigmoid(Q)
            x_glu = P_with_rc * self.sigmoid(Q)
            x_tc_out = x_glu
        elif self.act_func == "Sigmoid":
            # Temporal Convolution Layer (Sigmoid)
            x_sigmoid = self.sigmoid(x_conv)
            x_tc_out = x_sigmoid
        elif self.act_func == "ReLU":
            # Temporal Convolution Layer (ReLU)
            x_relu = self.relu(x_conv + x_input)
            x_tc_out = x_relu
        elif self.act_func == "Linear":
            # Temporal Convolution Layer (Linear)
            x_linear = x_conv
            x_tc_out = x_linear
        else:
            raise ValueError(f'ERROR: activation function "{act_func}" is not defined.')
        return x_tc_out

class ChebConv(nn.Module):
    def __init__(self, c_in, c_out, Ks, chebconv_filter, enable_bias):
        super(ChebConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.Ks = Ks
        self.chebconv_filter = chebconv_filter
        self.enable_bias = enable_bias
        self.weight = nn.Parameter(torch.FloatTensor(Ks, c_in, c_out))
        if self.enable_bias == True:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter('bias', None)
        self.initialize_parameters()

    def initialize_parameters(self):
        #init.xavier_uniform_(self.weight)
        init.kaiming_uniform_(self.weight)

        if self.bias is not None:
            _out_feats_bias = self.bias.size(0)
            stdv_b = 1. / math.sqrt(_out_feats_bias)
            init.uniform_(self.bias, -stdv_b, stdv_b)

    def forward(self, x):
        batch_size, c_in, T, n_vertex = x.shape

        x_before_first_mul = x.reshape(-1, c_in)
        x_first_mul = torch.mm(x_before_first_mul, self.weight.reshape(c_in, -1)).reshape(n_vertex * self.Ks, -1)
        x_second_mul = torch.spmm(self.chebconv_filter, x_first_mul).reshape(-1, self.c_out)

        if self.bias is not None:
            x_chebconv = x_second_mul + self.bias
        else:
            x_chebconv = x_second_mul
        return x_chebconv

class GCNConv(nn.Module):
    def __init__(self, c_in, c_out, gcnconv_filter, enable_bias):
        super(GCNConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.gcnconv_filter = gcnconv_filter
        self.enable_bias = enable_bias
        self.weight = nn.Parameter(torch.FloatTensor(c_in, c_out))
        if enable_bias == True:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter('bias', None)
        self.initialize_parameters()

    def initialize_parameters(self):
        #init.xavier_uniform_(self.weight)
        init.kaiming_uniform_(self.weight)

        if self.bias is not None:
            _out_feats_bias = self.bias.size(0)
            stdv_b = 1. / math.sqrt(_out_feats_bias)
            init.uniform_(self.bias, -stdv_b, stdv_b)

    def forward(self, x):
        batch_size, c_in, T, n_vertex = x.shape

        x_before_first_mul = x.reshape(-1, c_in)
        x_first_mul = torch.mm(x_before_first_mul, self.weight).reshape(n_vertex, -1)
        x_second_mul = torch.spmm(self.gcnconv_filter, x_first_mul).reshape(-1, self.c_out)

        if self.bias is not None:
            x_gcnconv_out = x_second_mul + self.bias
        else:
            x_gcnconv_out = x_second_mul
        return x_gcnconv_out

class SpatialConvLayer(nn.Module):
    def __init__(self, Ks, c_in, c_out, graph_conv_type, graph_conv_filter):
        super(SpatialConvLayer, self).__init__()
        self.Ks = Ks
        self.c_in = c_in
        self.c_out = c_out
        self.graph_conv_type = graph_conv_type
        self.graph_conv_filter = graph_conv_filter
        self.enable_bias = True
        if self.graph_conv_type == "chebconv":
            self.chebconv = ChebConv(c_in, c_out, self.Ks, self.graph_conv_filter, self.enable_bias)
        elif self.graph_conv_type == "gcnconv":
            self.gcnconv = GCNConv(c_in, c_out, self.graph_conv_filter, self.enable_bias)

    def forward(self, x):
        batch_size, c_in, T, n_vertex = x.shape
        x_gc_in = x
        if self.graph_conv_type == "chebconv":
            x_gc_out = self.chebconv(x_gc_in)
        elif self.graph_conv_type == "gcnconv":
            x_gc_out = self.gcnconv(x_gc_in)
        x_sc = x_gc_out.reshape(-1, T, n_vertex, self.c_out).permute(0, 3, 1, 2).contiguous()
        x_sc_with_rc = x_sc[:, : self.c_out, :, :] + x_gc_in.contiguous()
        x_sc_out = x_sc_with_rc
        return x_sc_out

class STConvBlock(nn.Module):
    # each STConvBlock contains 'TSTN' structure
    # T: Temporal Convolution Layer (GLU)
    # S: Spitial Convolution Layer (ChebConv or GCNConv)
    # T: Temporal Convolution Layer (ReLU)
    # N: Layer Normolization

    def __init__(self, Kt, Ks, n_vertex, channel, gc, graph_conv_filter, drop_prob):
        super(STConvBlock, self).__init__()
        self.tmp_conv1 = TemporalConvLayer(Kt, channel[0], channel[1], "GLU")
        self.spat_conv = SpatialConvLayer(Ks, channel[1], channel[1], gc, graph_conv_filter)
        self.tmp_conv2 = TemporalConvLayer(Kt, channel[1], channel[2], "ReLU")
        self.relu = nn.ReLU()
        #self.ln1 = nn.LayerNorm([n_vertex, channel[1]])
        self.ln2 = nn.LayerNorm([n_vertex, channel[2]])
        self.spat_dropout = nn.Dropout2d(drop_prob)

    def forward(self, x):
        x_tmp_conv1 = self.tmp_conv1(x)
        x_spat_conv = self.spat_conv(x_tmp_conv1)
        x_relu = self.relu(x_spat_conv)
        x_tmp_conv2 = self.tmp_conv2(x_relu)
        x_ln2 = self.ln2(x_tmp_conv2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_spat_do = self.spat_dropout(x_ln2)
        x_st_conv_out = x_spat_do
        return x_st_conv_out

class OutputLayer(nn.Module):
    # output layer contains 'TNTF' structure
    # T: Temporal Convolution Layer (GLU)
    # N: Layer Normalization
    # T: Temporal Convolution Layer (Sigmoid)
    # F: Fully-Connected Layer

    def __init__(self, c_in, T, n_vertex):
        super(OutputLayer, self).__init__()
        self.tmp_conv1 = TemporalConvLayer(T, c_in, c_in, "GLU")
        self.ln = nn.LayerNorm([n_vertex, c_in])
        self.tmp_conv2 = TemporalConvLayer(1, c_in, c_in, "Sigmoid")
        self.fc = nn.Conv2d(c_in, 1, 1)

    def forward(self, x):
        x_tc1 = self.tmp_conv1(x)
        x_ln = self.ln(x_tc1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_tc2 = self.tmp_conv2(x_ln)
        x_fc = self.fc(x_tc2)
        x_out = x_fc
        return x_out

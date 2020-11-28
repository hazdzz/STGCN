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
        else: # self.c_in == self.c_out
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
        self.leaky_relu = nn.LeakyReLU()

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
        elif self.act_func == "LeakyReLU":
            # Temporal Convolution Layer (LeakyReLU)
            x_leaky_relu = self.leaky_relu(x_conv + x_input)
            x_tc_out = x_leaky_relu
        elif self.act_func == "Linear":
            # Temporal Convolution Layer (Linear)
            x_linear = x_conv
            x_tc_out = x_linear
        else:
            raise ValueError(f'ERROR: activation function "{act_func}" is not defined.')
        return x_tc_out

class GraphConvolution_CPA(nn.Module):
    def __init__(self, c_in, c_out, Ks, gc_cpa_kernel, enable_bias):
        super(GraphConvolution_CPA, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.Ks = Ks
        self.gc_cpa_kernel = gc_cpa_kernel
        self.enable_bias = enable_bias
        self.weight = nn.Parameter(torch.FloatTensor(Ks * c_in, c_out))
        if self.enable_bias == True:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter('bias', None)
        self.initialize_parameters()

    def initialize_parameters(self):
        init.xavier_uniform_(self.weight)
        #init.kaiming_uniform_(self.weight)

        if self.bias is not None:
            _out_feats_bias = self.bias.size(0)
            stdv_b = 1. / math.sqrt(_out_feats_bias)
            init.uniform_(self.bias, -stdv_b, stdv_b)

    def forward(self, x):
        _, n_vertex, c_in = x.shape
        x_before_first_mul = x.permute(0, 2, 1).reshape(-1, n_vertex)
        x_first_mul = torch.matmul(x_before_first_mul, self.gc_cpa_kernel).reshape(-1, c_in, self.Ks, n_vertex)
        x_before_second_mul = x_first_mul.permute(0, 3, 1, 2).reshape(-1, c_in * self.Ks)
        x_second_mul = torch.matmul(x_before_second_mul, self.weight).reshape(-1, n_vertex, self.c_out)
        if self.bias is not None:
            x_gc_cpa = x_second_mul + self.bias
        else:
            x_gc_cpa = x_second_mul
        return x_gc_cpa

class GraphConvolution_LWL(nn.Module):
    def __init__(self, c_in, c_out, gc_lwl_kernel, enable_bias):
        super(GraphConvolution_LWL, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.gc_lwl_kernel = gc_lwl_kernel
        self.enable_bias = enable_bias
        self.weight = nn.Parameter(torch.FloatTensor(c_in, c_out))
        if enable_bias == True:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter('bias', None)
        self.initialize_parameters()

    def initialize_parameters(self):
        #_out_feats_weight = self.weight.size(1)
        #stdv_w = 1. / math.sqrt(_out_feats_weight)
        #init.uniform_(self.weight, -stdv_w, stdv_w)
        
        init.xavier_uniform_(self.weight)
        #init.kaiming_uniform_(self.weight)

        if self.bias is not None:
            _out_feats_bias = self.bias.size(0)
            stdv_b = 1. / math.sqrt(_out_feats_bias)
            init.uniform_(self.bias, -stdv_b, stdv_b)

    def forward(self, x):
        _, n_vertex, c_in = x.shape

        # The STGCN author's way
        #x_before_first_mul = x.permute(0, 2, 1).reshape(-1, n_vertex)
        #x_first_mul = torch.matmul(x_before_first_mul, self.gc_lwl_kernel).reshape(-1, c_in, 1, n_vertex)
        #x_before_second_mul = x_first_mul.permute(0, 3, 1, 2).reshape(-1, c_in)
        #x_second_mul = torch.matmul(x_before_second_mul, self.weight).reshape(-1, n_vertex, self.c_out)

        # There are the code from GraphConvolution of GCN
        #support = torch.mm(input, self.weight)
        #output = torch.spmm(adj, support)
        # The GCN author's way
        x_before_first_mul = x.reshape(-1, c_in)
        x_first_mul = torch.mm(x_before_first_mul, self.weight).reshape(n_vertex, -1)
        x_second_mul = torch.spmm(self.gc_lwl_kernel, x_first_mul).reshape(-1, self.c_out)

        if self.bias is not None:
            x_gc_lwl = x_second_mul + self.bias
        else:
            x_gc_lwl = x_second_mul
        return x_gc_lwl

class SpatialGraphConvLayer(nn.Module):
    def __init__(self, Ks, c_in, c_out, gc, graph_conv_kernel):
        super(SpatialGraphConvLayer, self).__init__()
        self.Ks = Ks
        self.c_in = c_in
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        self.gc = gc
        self.graph_conv_kernel = graph_conv_kernel
        self.enbale_bias = True
        if self.gc == "gc_cpa":
            self.gc_cpa = GraphConvolution_CPA(c_in, c_out, self.Ks, self.graph_conv_kernel, self.enbale_bias)
        elif self.gc == "gc_lwl":
            self.gc_lwl = GraphConvolution_LWL(c_in, c_out, self.graph_conv_kernel, self.enbale_bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size, c_in, T, n_vertex = x.shape
        x_input = self.align(x)
        x_gc_input = x.permute(0, 2, 3, 1).reshape(-1, n_vertex, c_in)
        if self.gc == "gc_cpa":
            x_gc_output = self.gc_cpa(x_gc_input)
        elif self.gc == "gc_lwl":
            x_gc_output = self.gc_lwl(x_gc_input)
        x_sgc = x_gc_output.reshape(-1, T, n_vertex, self.c_out).permute(0, 3, 1, 2).contiguous()
        x_sgc_with_rc = x_sgc[:, : self.c_out, :, :] + x_input.contiguous()
        x_sgc_output = self.relu(x_sgc_with_rc)
        return x_sgc_output

class STConvBlock(nn.Module):
    # each STConvBlock contains 'TSTN' structure
    # T: Temporal Convolution Layer (GLU)
    # S: Spitial Graph Convolution Layer (GC_CPA or GC_LWL)
    # T: Temporal Convolution Layer (ReLU)
    # N: Layer Normolization

    def __init__(self, Kt, Ks, n_vertex, channel, gc, graph_conv_kernel, drop_prob):
        super(STConvBlock, self).__init__()
        self.tmp_conv1 = TemporalConvLayer(Kt, channel[0], channel[1], "GLU")
        self.sg_conv = SpatialGraphConvLayer(Ks, channel[1], channel[1], gc, graph_conv_kernel)
        self.tmp_conv2 = TemporalConvLayer(Kt, channel[1], channel[2], "ReLU")
        self.ln = nn.LayerNorm([n_vertex, channel[2]])
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        x_tmp_conv1 = self.tmp_conv1(x)
        x_sg_conv = self.sg_conv(x_tmp_conv1)
        x_tmp_conv2 = self.tmp_conv2(x_sg_conv)
        x_ln = self.ln(x_tmp_conv2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_do = self.dropout(x_ln)
        return x_do

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
        return x_fc

class STGCN_GC_CPA(nn.Module):
    # STGCN(GC_CPA) contains 'TSTN TSTN TNTF' structure
        
    # T: Temporal Convolution Layer (GLU)
    # S: Spitial Graph Convolution Layer (CPA)
    # T: Temporal Convolution Layer (ReLU)
    # N: Layer Normolization

    # T: Temporal Convolution Layer (GLU)
    # S: Spitial Graph Convolution Layer (CPA)
    # T: Temporal Convolution Layer (ReLU)
    # N: Layer Normolization

    # T: Temporal Convolution Layer (GLU)
    # N: Layer Normalization
    # T: Temporal Convolution Layer (Sigmoid)
    # F: Fully-Connected Layer

    def __init__(self, Kt, Ks, blocks, T, n_vertex, gc, gc_cpa_kernel, drop_prob):
        super(STGCN_GC_CPA, self).__init__()
        self.st_block1 = STConvBlock(Kt, Ks, n_vertex, blocks[0], gc, gc_cpa_kernel, drop_prob)
        self.st_block2 = STConvBlock(Kt, Ks, n_vertex, blocks[1], gc, gc_cpa_kernel, drop_prob)
        Ko = T - len(blocks) * 2 * (Kt - 1)
        if Ko > 1:
            self.output = OutputLayer(blocks[-1][-1], Ko, n_vertex)
        else:
            raise ValueError(f'ERROR: kernel size Ko must be greater than 1, but received "{Ko}".')

    def forward(self, x):
        x_stb1 = self.st_block1(x)
        x_stb2 = self.st_block2(x_stb1)
        x_out = self.output(x_stb2)
        return x_out

class STGCN_GC_LWL(nn.Module):
    # STGCN(GC_LWL) contains 'TSTN TSTN TNTF' structure
        
    # T: Temporal Convolution Layer (GLU)
    # S: Spitial Graph Convolution Layer (LWL)
    # T: Temporal Convolution Layer (ReLU)
    # N: Layer Normolization

    # T: Temporal Convolution Layer (GLU)
    # S: Spitial Graph Convolution Layer (LWL)
    # T: Temporal Convolution Layer (ReLU)
    # N: Layer Normolization

    # T: Temporal Convolution Layer (GLU)
    # N: Layer Normalization
    # T: Temporal Convolution Layer (Sigmoid)
    # F: Fully-Connected Layer

    def __init__(self, Kt, Ks, blocks, T, n_vertex, gc, gc_lwl_kernel, drop_prob):
        super(STGCN_GC_LWL, self).__init__()
        self.st_block1 = STConvBlock(Kt, Ks, n_vertex, blocks[0], gc, gc_lwl_kernel, drop_prob)
        self.st_block2 = STConvBlock(Kt, Ks, n_vertex, blocks[1], gc, gc_lwl_kernel, drop_prob)
        Ko = T - len(blocks) * 2 * (Kt - 1)
        if Ko > 1:
            self.output = OutputLayer(blocks[-1][-1], Ko, n_vertex)
        else:
            raise ValueError(f'ERROR: kernel size Ko must be greater than 1, but received "{Ko}".')

    def forward(self, x):
        x_stb1 = self.st_block1(x)
        x_stb2 = self.st_block2(x_stb1)
        x_out = self.output(x_stb2)
        return x_out

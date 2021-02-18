import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(in_channels=self.c_in, out_channels=self.c_out, kernel_size=(1, 1))

    def forward(self, x):
        if self.c_in > self.c_out:
            x_align = self.align_conv(x)
        elif self.c_in < self.c_out:
            batch_size, c_in, timestep, n_vertex = x.shape
            x_align = torch.cat([x, torch.zeros([batch_size, self.c_out - self.c_in, timestep, n_vertex]).to(x)], dim=1)
        else:
            x_align = x
        return x_align

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

    def __init__(self, Kt, c_in, c_out, n_vertex, act_func, enable_gated_act_func):
        super(TemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.n_vertex = n_vertex
        self.act_func = act_func
        self.enable_gated_act_func = enable_gated_act_func
        self.align = Align(self.c_in, self.c_out)
        if self.enable_gated_act_func == True:
            self.causal_conv = nn.Conv2d(in_channels=self.c_in, out_channels=2 * self.c_out, kernel_size=(self.Kt, 1), dilation=1)
        else:
            self.causal_conv = nn.Conv2d(in_channels=self.c_in, out_channels=self.c_out, kernel_size=(self.Kt, 1), dilation=1)
        self.linear = nn.Linear(self.n_vertex, self.n_vertex)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
        self.elu = nn.ELU()
        self.softplus = nn.Softplus()
        self.softsign = nn.Softsign()

    def forward(self, x):   
        x_in = self.align(x)[:, :, self.Kt - 1:, :]
        x_causal_conv = self.causal_conv(x)

        if self.enable_gated_act_func == True:
            x_p = x_causal_conv[:, : self.c_out, :, :]
            x_q = x_causal_conv[:, -self.c_out:, :, :]

            # Temporal Convolution Layer (GLU)
            if self.act_func == "glu":
                # GLU was first purposed in
                # Language Modeling with Gated Convolutional Networks
                # https://arxiv.org/abs/1612.08083
                # Input tensor X was split by a certain dimension into tensor X_a and X_b
                # In original paper, GLU as Linear(X_a) ⊙ Sigmoid(Linear(X_b))
                # However, in PyTorch, GLU as X_a ⊙ Sigmoid(X_b)
                # https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.glu
                # Because in original paper, the representation of GLU and GTU are ambiguous
                # So, it is arguable which one version is correct

                # (x_p + x_in) ⊙ Sigmoid(x_q)
                x_glu = torch.mul((x_p + x_in), self.sigmoid(x_q))
                x_tc_out = x_glu

            # Temporal Convolution Layer (GTU)
            elif self.act_func == "gtu":
                # Tanh(x_p + x_in) ⊙ Sigmoid(x_q)
                x_gtu = torch.mul(self.tanh(x_p + x_in), self.sigmoid(x_q))
                x_tc_out = x_gtu

            else:
                raise ValueError(f'ERROR: activation function "{self.act_func}" is not defined.')

        else:

            # Temporal Convolution Layer (Linear)
            if self.act_func == "linear":
                x_linear = self.linear(x_causal_conv + x_in)
                x_tc_out = x_linear
            
            # Temporal Convolution Layer (Sigmoid)
            elif self.act_func == "sigmoid":
                x_sigmoid = self.sigmoid(x_causal_conv + x_in)
                x_tc_out = x_sigmoid

            # Temporal Convolution Layer (Tanh)
            elif self.act_func == "tanh":
                x_tanh = self.tanh(x_causal_conv + x_in)
                x_tc_out = x_tanh

            # Temporal Convolution Layer (ReLU)
            elif self.act_func == "relu":
                x_relu = self.relu(x_causal_conv + x_in)
                x_tc_out = x_relu
        
            # Temporal Convolution Layer (LeakyReLU)
            elif self.act_func == "leakyrelu":
                x_leakyrelu = self.leakyrelu(x_causal_conv + x_in)
                x_tc_out = x_leakyrelu

            # Temporal Convolution Layer (ELU)
            elif self.act_func == "elu":
                x_elu = self.elu(x_causal_conv + x_in)
                x_tc_out = x_elu

            # Temporal Convolution Layer (Softplus)
            elif self.act_func == "softplus":
                x_softplus = self.softplus(x_causal_conv + x_in)
                x_tc_out = x_softplus

            # Temporal Convolution Layer (Softsign)
            elif self.act_func == "softsign":
                x_softsign = self.softsign(x_causal_conv + x_in)
                x_tc_out = x_softsign

            else:
                raise ValueError(f'ERROR: activation function "{self.act_func}" is not defined.')
        
        return x_tc_out

class ChebConv(nn.Module):
    def __init__(self, c_in, c_out, Ks, chebconv_matrix_list, enable_bias):
        super(ChebConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.Ks = Ks
        self.chebconv_matrix_list = chebconv_matrix_list
        self.enable_bias = enable_bias
        self.weight = nn.Parameter(torch.FloatTensor(Ks, c_in, c_out))
        if self.enable_bias == True:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter('bias', None)
        self.initialize_parameters()

    def initialize_parameters(self):
        # For Sigmoid or Tanh
        #init.xavier_uniform_(self.weight)
        # For ReLU or Leaky ReLU
        init.kaiming_uniform_(self.weight)

        if self.bias is not None:
            _out_feats_bias = self.bias.size(0)
            stdv_b = 1. / math.sqrt(_out_feats_bias)
            init.uniform_(self.bias, -stdv_b, stdv_b)

    def forward(self, x):
        batch_size, c_in, T, n_vertex = x.shape

        x_before_first_mul = x.reshape(-1, c_in)
        x_first_mul = torch.mm(x_before_first_mul, self.weight.reshape(c_in, -1)).reshape(n_vertex * self.Ks, -1)
        x_second_mul = torch.mm(self.chebconv_matrix_list, x_first_mul).reshape(-1, self.c_out)

        if self.bias is not None:
            x_chebconv = x_second_mul + self.bias
        else:
            x_chebconv = x_second_mul
        
        return x_chebconv

class GCNConv(nn.Module):
    def __init__(self, c_in, c_out, gcnconv_matrix, enable_bias):
        super(GCNConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.gcnconv_matrix = gcnconv_matrix
        self.enable_bias = enable_bias
        self.weight = nn.Parameter(torch.FloatTensor(c_in, c_out))
        if enable_bias == True:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter('bias', None)
        self.initialize_parameters()

    def initialize_parameters(self):
        # For Sigmoid or Tanh
        #init.xavier_uniform_(self.weight)
        # For ReLU or Leaky ReLU
        init.kaiming_uniform_(self.weight)

        if self.bias is not None:
            _out_feats_bias = self.bias.size(0)
            stdv_b = 1. / math.sqrt(_out_feats_bias)
            init.uniform_(self.bias, -stdv_b, stdv_b)

    def forward(self, x):
        batch_size, c_in, T, n_vertex = x.shape

        x_before_first_mul = x.reshape(-1, c_in)
        x_first_mul = torch.mm(x_before_first_mul, self.weight).reshape(n_vertex, -1)
        x_second_mul = torch.spmm(self.gcnconv_matrix, x_first_mul).reshape(-1, self.c_out)

        if self.bias is not None:
            x_gcnconv_out = x_second_mul + self.bias
        else:
            x_gcnconv_out = x_second_mul
        
        return x_gcnconv_out

class GraphConvLayer(nn.Module):
    def __init__(self, Ks, c_in, c_out, graph_conv_type, graph_conv_matrix):
        super(GraphConvLayer, self).__init__()
        self.Ks = Ks
        self.c_in = c_in
        self.c_out = c_out
        self.align = Align(self.c_in, self.c_out)
        self.graph_conv_type = graph_conv_type
        self.graph_conv_matrix = graph_conv_matrix
        self.enable_bias = True
        if self.graph_conv_type == "chebconv":
            self.chebconv = ChebConv(self.c_out, self.c_out, self.Ks, self.graph_conv_matrix, self.enable_bias)
        elif self.graph_conv_type == "gcnconv":
            self.gcnconv = GCNConv(self.c_out, self.c_out, self.graph_conv_matrix, self.enable_bias)

    def forward(self, x):
        x_gc_in = self.align(x)
        batch_size, c_in, T, n_vertex = x_gc_in.shape
        if self.graph_conv_type == "chebconv":
            x_gc_out = self.chebconv(x_gc_in)
        elif self.graph_conv_type == "gcnconv":
            x_gc_out = self.gcnconv(x_gc_in)
        x_gc_with_rc = x_gc_out.reshape(batch_size, self.c_out, T, n_vertex).contiguous() + x_gc_in.contiguous()
        x_gc_out = x_gc_with_rc
        return x_gc_out

class STConvBlock(nn.Module):
    # STConv Block contains 'TNSATND' structure
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebConv or GCNConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    def __init__(self, Kt, Ks, n_vertex, last_block_channel, channel, gated_act_func, graph_conv_type, graph_conv_matrix, drop_rate):
        super(STConvBlock, self).__init__()
        self.Kt = Kt
        self.Ks = Ks
        self.n_vertex = n_vertex
        self.last_block_channel = last_block_channel
        self.channel = channel
        self.gated_act_func = gated_act_func
        self.enable_gated_act_func = True
        self.graph_conv_type = graph_conv_type
        self.graph_conv_matrix = graph_conv_matrix
        self.drop_rate = drop_rate
        self.tmp_conv1 = TemporalConvLayer(self.Kt, self.last_block_channel, self.channel[0], self.n_vertex, self.gated_act_func, self.enable_gated_act_func)
        self.spat_conv = GraphConvLayer(self.Ks, self.channel[0], self.channel[1], self.graph_conv_type, self.graph_conv_matrix)
        self.tmp_conv2 = TemporalConvLayer(self.Kt, self.channel[1], self.channel[2], self.n_vertex, self.gated_act_func, self.enable_gated_act_func)
        self.tc2_ln = nn.LayerNorm([self.n_vertex, self.channel[2]])
        self.relu = nn.ReLU()
        #self.leakyrelu = nn.LeakyReLU()
        #self.elu = nn.ELU()
        self.do = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        x_tmp_conv1 = self.tmp_conv1(x)
        x_spat_conv = self.spat_conv(x_tmp_conv1)
        x_relu = self.relu(x_spat_conv)
        x_tmp_conv2 = self.tmp_conv2(x_relu)
        x_tc2_ln = self.tc2_ln(x_tmp_conv2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_do = self.do(x_tc2_ln)
        x_st_conv_out = x_do
        return x_st_conv_out

class OutputBlock(nn.Module):
    # Output block contains 'TNFF' structure
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer

    def __init__(self, Ko, last_block_channel, channel, end_channel, n_vertex, gated_act_func, drop_rate):
        super(OutputBlock, self).__init__()
        self.Ko = Ko
        self.last_block_channel = last_block_channel
        self.channel = channel
        self.end_channel = end_channel
        self.n_vertex = n_vertex
        self.gated_act_func = gated_act_func
        self.enable_gated_act_func = True
        self.drop_rate = drop_rate
        self.tmp_conv1 = TemporalConvLayer(self.Ko, self.last_block_channel, self.channel[0], self.n_vertex, self.gated_act_func, self.enable_gated_act_func)
        self.fc1 = nn.Linear(self.channel[0], self.channel[1])
        self.fc2 = nn.Linear(self.channel[1], self.end_channel)
        self.tc1_ln = nn.LayerNorm([self.n_vertex, self.channel[0]])
        self.sigmoid = nn.Sigmoid()
        #self.tanh = nn.Tanh()
        #self.relu = nn.ReLU()
        #self.do = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        x_tc1 = self.tmp_conv1(x)
        x_tc1_ln = self.tc1_ln(x_tc1.permute(0, 2, 3, 1))
        x_fc1 = self.fc1(x_tc1_ln)
        x_sigmoid = self.sigmoid(x_fc1)
        x_fc2 = self.fc2(x_sigmoid).permute(0, 3, 1, 2)
        x_out = x_fc2
        return x_out
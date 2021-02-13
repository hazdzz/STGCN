# Spatio-Temporal Graph Convolutional Networks
[![issues](https://img.shields.io/github/issues/hazdzz/STGCN)](https://github.com/hazdzz/STGCN/issues)
[![forks](https://img.shields.io/github/forks/hazdzz/STGCN)](https://github.com/hazdzz/STGCN/network/members)
[![stars](https://img.shields.io/github/stars/hazdzz/STGCN)](https://github.com/hazdzz/STGCN/stargazers)
[![License](https://img.shields.io/github/license/hazdzz/STGCN)](./LICENSE)

## About
The PyTorch version of STGCN implemented by the paper *Spatio-Temporal Graph Convolutional Networks:
A Deep Learning Framework for Traffic Forecasting* with tons of bugs fixed.

## Paper
https://arxiv.org/abs/1709.04875

## Related Works
1. TCN: [*An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*](https://arxiv.org/abs/1803.01271)
2. ChebyNet: [*Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering*](https://arxiv.org/abs/1606.09375)
3. GCN: [*Semi-Supervised Classification with Graph Convolutional Networks*](https://arxiv.org/abs/1609.02907)

## Related Code
1. TCN: https://github.com/locuslab/TCN
2. ChebyNet: https://github.com/mdeff/cnn_graph
3. GCN: https://github.com/tkipf/pygcn

## Model structure
<img src="./figure/stgcn_model_structure.png" style="zoom:100%" />

## Differents of code between mine and author's
1. Fix tons of bugs 
2. Add Early Stopping approach
3. Add Dropout approach
4. Offer a different set of hyperparameters
5. Offer config files for two different categories graph convolution

## The result for road traffic prediction on dataset PeMSD7(M)(15/30/45 mins)
### 15 mins (H=3)
|  Model(paper)  |  Model(code)  |  Laplacian matrix type  |  Gated activation function  |  MAE  |  MAPE  |  RMSE  |
|  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |
|  STGCN(Cheb)  |  STGCN_ChebConv(Ks=3, Kt=3)  |  sym  |  GLU  |  2.196439  |  5.120554%  |  3.942155  |
|  STGCN(1<sup>st</sup>)  |  STGCN_GCNConv(Kt=3)  |  sym  |  GLU  |  2.200761  |  5.101906%  |  3.937438  |

### 30 mins (H=6)
|  Model(paper)  |  Model(code)  |  Laplacian matrix type  |  Gated activation function  |  MAE  |  MAPE  |  RMSE  |
|  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |
|  STGCN(Cheb)  |  STGCN_ChebConv(Ks=3, Kt=3)  |  sym  |  GLU  |  2.908268  |  7.004069%  |  5.287514  |
|  STGCN(1<sup>st</sup>)  |  STGCN_GCNConv(Kt=3)  |  sym  |  GLU  |  2.831076  |  6.861572%  |  5.175758  |

### 45 mins (H=9)
|  Model(paper)  |  Model(code)  |  Laplacian matrix type  |  Gated activation function  |  MAE  |  MAPE  |  RMSE  |
|  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |
|  STGCN(Cheb)  |  STGCN_ChebConv(Ks=3, Kt=3)  |  sym  |  GLU  |  3.312900  |  8.111004%  |  6.018192  |
|  STGCN(1<sup>st</sup>)  |  STGCN_GCNConv(Kt=3)  |  sym  |  GLU  |  3.195088  |  7.912600%  |  5.845828  |

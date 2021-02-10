# Spatio-Temporal Graph Convolutional Networks
[![issues](https://img.shields.io/github/issues/hazdzz/STGCN)](https://github.com/hazdzz/STGCN/issues)
[![forks](https://img.shields.io/github/forks/hazdzz/STGCN)](https://github.com/hazdzz/STGCN/network/members)
[![stars](https://img.shields.io/github/stars/hazdzz/STGCN)](https://github.com/hazdzz/STGCN/stargazers)
[![License](https://img.shields.io/github/license/hazdzz/STGCN)](./LICENSE)

The PyTorch version of STGCN implemented by the paper *Spatio-Temporal Graph Convolutional Networks:
A Deep Learning Framework for Traffic Forecasting* with tons of bugs fixed

## Paper
https://arxiv.org/abs/1709.04875

## Model structure
<img src="./figure/stgcn_model_structure.png" style="zoom:100%" />

## Differents of code between mine and author's
1. Fix tons of bugs 
2. Add Early Stopping approach
3. Add Dropout approach
4. Offer a different set of hyperparameters which untuned
5. Offer config files for two different categories graph convolution

## The result for road traffic prediction on dataset PeMSD7(M)(15/30/45 mins)
### 15 mins (H=3)
|  Model(paper)  |  Model(code)  |  Laplacian matrix type  |  Gated activation function  |  MAE  |  MAPE  |  RMSE  |
|  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |
|  STGCN(Cheb)  |  STGCN_ChebConv(Ks=3, Kt=3)  |  sym  |  GLU  |  2.203283  |  5.159329%  |  3.944862  |
|  STGCN(1<sup>st</sup>)  |  STGCN_GCNConv(Kt=3)  |  sym  |  GLU  |  2.191923  |  5.097812%  |  3.940933  |

### 30 mins (H=6)
|  Model(paper)  |  Model(code)  |  Laplacian matrix type  |  Gated activation function  |  MAE  |  MAPE  |  RMSE  |
|  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |
|  STGCN(Cheb)  |  STGCN_ChebConv(Ks=3, Kt=3)  |  sym  |  GLU  |  2.898282  |  7.175031%  |  5.300563  |
|  STGCN(1<sup>st</sup>)  |  STGCN_GCNConv(Kt=3)  |  sym  |  GLU  |  2.857253  |  6.969964%  |  5.204885  |

### 45 mins (H=9)
|  Model(paper)  |  Model(code)  |  Laplacian matrix type  |  Gated activation function  |  MAE  |  MAPE  |  RMSE  |
|  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |
|  STGCN(Cheb)  |  STGCN_ChebConv(Ks=3, Kt=3)  |  sym  |  GLU  |  3.224847  |  8.084734%  |  5.938307  |
|  STGCN(1<sup>st</sup>)  |  STGCN_GCNConv(Kt=3)  |  sym  |  GLU  |  3.220803  |  8.033510%  |  5.877929  |

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
|  STGCN(Cheb)  |  STGCN_ChebConv(Ks=3, Kt=3)  |  sym  |  GLU  |  2.258984  |  5.330609%  |  4.025423  |
|  STGCN(1<sup>st</sup>)  |  STGCN_GCNConv(Kt=3)  |  sym  |  GLU  |  2.230498  |  5.249552%  |  3.983621  |

### 30 mins (H=6)
|  Model(paper)  |  Model(code)  |  Laplacian matrix type  |  Gated activation function  |  MAE  |  MAPE  |  RMSE  |
|  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |
|  STGCN(Cheb)  |  STGCN_ChebConv(Ks=3, Kt=3)  |  sym  |  GLU  |  2.877363  |  7.028274%  |  5.256138  |
|  STGCN(1<sup>st</sup>)  |  STGCN_GCNConv(Kt=3)  |  sym  |  GLU  |  2.902291  |  7.165167%  |  5.302092  |

### 45 mins (H=9)
|  Model(paper)  |  Model(code)  |  Laplacian matrix type  |  Gated activation function  |  MAE  |  MAPE  |  RMSE  |
|  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |
|  STGCN(Cheb)  |  STGCN_ChebConv(Ks=3, Kt=3)  |  sym  |  GLU  |  3.196036  |  8.048447%  |  5.904350  |
|  STGCN(1<sup>st</sup>)  |  STGCN_GCNConv(Kt=3)  |  sym  |  GLU  |  3.248762  |  8.097897%  |  5.914715  |

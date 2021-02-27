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

## Related works
1. TCN: [*An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*](https://arxiv.org/abs/1803.01271)
2. GLU and GTU: [*Language Modeling with Gated Convolutional Networks*](https://arxiv.org/abs/1612.08083)
3. ChebyNet: [*Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering*](https://arxiv.org/abs/1606.09375)
4. GCN: [*Semi-Supervised Classification with Graph Convolutional Networks*](https://arxiv.org/abs/1609.02907)

## Related code
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

## The result for road traffic prediction on dataset PeMSD7(M) (15/30/45 mins)
### 15 mins (H=3)
| Model (paper) | Model (code) | Laplacian matrix type | Gated activation function | MAE | MAPE | RMSE |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| STGCN (Cheb) | STGCN_ChebConv (Ks=3, Kt=3) | sym | GLU | 2.196439 | 5.120554% | 3.942155 |
| | STGCN_ChebConv (Ks=3, Kt=3) | sym | GTU | 2.192645 | 5.097086% | 3.938440 |
| | STGCN_ChebConv (Ks=3, Kt=3) | rw | GLU | 2.194124 | 5.118343% | 3.946593 |
| | STGCN_ChebConv (Ks=3, Kt=3) | rw | GTU | 2.193308 | 5.103092% | 3.941704 |
| STGCN (1<sup>st</sup>) | STGCN_GCNConv (Kt=3) | sym | GLU | 2.200761 | 5.101906% | 3.937438 |
| | STGCN_GCNConv (Kt=3) | sym | GTU | 2.177197 | 5.054439% | 3.932912 |
| | STGCN_GCNConv (Kt=3) | rw | GLU | 2.197893 | 5.093305% | 3.937244 |
| | STGCN_GCNConv (Kt=3) | rw | GTU | 2.176889 | 5.063183% | 3.933613 |

### 30 mins (H=6)
| Model (paper) | Model (code) | Laplacian matrix type | Gated activation function | MAE | MAPE | RMSE |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| STGCN (Cheb) | STGCN_ChebConv (Ks=3, Kt=3) | sym | GLU | 2.908268 | 7.004069% | 5.287514 |
| | STGCN_ChebConv (Ks=3, Kt=3) | sym | GTU | 2.850894 | 6.985739% | 5.245709 |
| | STGCN_ChebConv (Ks=3, Kt=3) | rw | GLU | 2.904776 | 6.965637% | 5.307057 |
| | STGCN_ChebConv (Ks=3, Kt=3) | rw | GTU | 2.905101 | 6.997626%| 5.276031 |
| STGCN (1<sup>st</sup>) | STGCN_GCNConv (Kt=3) | sym | GLU | 2.831076 | 6.861572% | 5.175758 |
| | STGCN_GCNConv (Kt=3) | sym | GTU | 2.841050 | 6.899995% | 5.229030 |
| | STGCN_GCNConv (Kt=3) | rw | GLU | 2.840195 | 6.862824% | 5.183369 |
| | STGCN_GCNConv (Kt=3) | rw | GTU | 2.834058 | 6.888992% | 5.217215 |

### 45 mins (H=9)
| Model (paper) | Model (code) | Laplacian matrix type | Gated activation function | MAE | MAPE | RMSE |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| STGCN (Cheb) | STGCN_ChebConv (Ks=3, Kt=3) | sym | GLU | 3.312900 | 8.111004% | 6.018192 |
| | STGCN_ChebConv (Ks=3, Kt=3) | sym | GTU | 3.364909 | 8.187228% | 6.028997 |
| | STGCN_ChebConv (Ks=3, Kt=3) | rw | GLU | 3.342183 | 8.320536% | 6.051046 |
| | STGCN_ChebConv (Ks=3, Kt=3) | rw | GTU | 3.360086 | 8.252135% | 6.044784 |
| STGCN (1<sup>st</sup>) | STGCN_GCNConv (Kt=3) | sym | GLU | 3.195088 | 7.912600% | 5.845828 |
| | STGCN_GCNConv (Kt=3) | sym | GTU | 3.164176 | 7.893403% | 5.846285 |
| | STGCN_GCNConv (Kt=3) | rw | GLU | 3.197024 | 7.926294%| 5.862989 |
| | STGCN_GCNConv (Kt=3) | rw | GTU | 3.166937 | 7.900064%| 5.842872 |

## The result for road traffic prediction on dataset METR-LA (15/30/60 mins)
### 15 mins (H=3)
| Model (paper) | Model (code) | Laplacian matrix type | Gated activation function | MAE | RMSE | WMAPE |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| STGCN (Cheb) | STGCN_ChebConv (Ks=3, Kt=3) | sym | GLU | 3.896181 | 8.180521 | 7.669819% |
| STGCN (1<sup>st</sup>) | STGCN_GCNConv (Kt=3) | sym | GLU | 3.703660 | 7.685864 | 7.290832% |

### 30 mins (H=6)
| Model (paper) | Model (code) | Laplacian matrix type | Gated activation function | MAE | RMSE | WMAPE |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| STGCN (Cheb) | STGCN_ChebConv (Ks=3, Kt=3) | sym | GLU | 5.415257 | 10.542593 | 10.661624% |
| STGCN (1<sup>st</sup>) | STGCN_GCNConv (Kt=3) | sym | GLU | 4.518740 | 8.863177 | 8.896550% |

### 60 mins (H=12)
| Model (paper) | Model (code) | Laplacian matrix type | Gated activation function | MAE | RMSE | WMAPE |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| STGCN (Cheb) | STGCN_ChebConv (Ks=3, Kt=3) | sym | GLU | 6.710720 | 12.415373 | 13.215713% |
| STGCN (1<sup>st</sup>) | STGCN_GCNConv (Kt=3) | sym | GLU | 5.997484 | 11.498759 | 11.811108% |

## The result for road traffic prediction on dataset PEMS-BAY (15/30/60 mins)
### 15 mins (H=3)
| Model (paper) | Model (code) | Laplacian matrix type | Gated activation function | MAE | RMSE | WMAPE |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| STGCN (Cheb) | STGCN_ChebConv (Ks=3, Kt=3) | sym | GLU | 1.515749 | 3.039355 | 2.439110% |
| STGCN (1<sup>st</sup>) | STGCN_GCNConv (Kt=3) | sym | GLU | 1.472308 | 2.987471 | 2.369206% |

### 30 mins (H=6)
| Model (paper) | Model (code) | Laplacian matrix type | Gated activation function | MAE | RMSE | WMAPE |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| STGCN (Cheb) | STGCN_ChebConv (Ks=3, Kt=3) | sym | GLU | 1.940019 | 4.057035 | 3.121926% |
| STGCN (1<sup>st</sup>) | STGCN_GCNConv (Kt=3) | sym | GLU | 1.910708 | 3.948517 | 3.074757% |

### 60 mins (H=12)
| Model (paper) | Model (code) | Laplacian matrix type | Gated activation function | MAE | RMSE | WMAPE |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| STGCN (Cheb) | STGCN_ChebConv (Ks=3, Kt=3) | sym | GLU | 2.316922 | 4.714354 | 3.728668% |
| STGCN (1<sup>st</sup>) | STGCN_GCNConv (Kt=3) | sym | GLU | 2.306092 | 4.701984 | 3.711238% |
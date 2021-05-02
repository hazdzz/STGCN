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

## Dataset
### Source
1. METR-LA: [DCRNN author's Google Drive](https://drive.google.com/file/d/1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC/view?usp=sharing)
2. PEMS-BAY: [DCRNN author's Google Drive](https://drive.google.com/file/d/1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq/view?usp=sharing)
3. PeMSD7(M): [STGCN author's GitHub repository](https://github.com/VeritasYin/STGCN_IJCAI-18/blob/master/data_loader/PeMS-M.zip)

### Preprocessing
Using the formula from [ChebyNet](https://arxiv.org/abs/1606.09375)：
<img src="./figure/Weighted Adjacency Matrix.png" style="zoom:100%" />

## Model structure
<img src="./figure/stgcn_model_structure.png" style="zoom:100%" />

## Differents of code between mine and author's
1. Fix tons of bugs 
2. Add Early Stopping approach
3. Add Dropout approach
4. Offer a different set of hyperparameters
5. Offer config files for two different categories graph convolution
6. Add datasets METR-LA and PEMS-BAY
7. Using a different data preprocessing method

## Requirements
To install requirements:
```console
pip3 install -r requirements.txt
```
## Experimental results
### METR-LA (15/30/60 mins) (train: val: test = 70: 15: 15)
#### 15 mins (H=3)
| Model (paper) | Model (code) | Laplacian matrix type | Gated activation function | MAE | RMSE | WMAPE |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| STGCN (Cheb) | STGCN_ChebConv (Ks=3, Kt=3) | sym | GLU | 3.825249 | 7.949693 | 7.530186% |
| STGCN (1<sup>st</sup>) | STGCN_GCNConv (Kt=3) | sym | GLU | 3.703660 | 7.685864 | 7.290832% |

#### 30 mins (H=6)
| Model (paper) | Model (code) | Laplacian matrix type | Gated activation function | MAE | RMSE | WMAPE |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| STGCN (Cheb) | STGCN_ChebConv (Ks=3, Kt=3) | sym | GLU | 4.789775 | 9.501917 | 9.430166% |
| STGCN (1<sup>st</sup>) | STGCN_GCNConv (Kt=3) | sym | GLU | 4.518740 | 8.863177 | 8.896550% |

#### 60 mins (H=12)
| Model (paper) | Model (code) | Laplacian matrix type | Gated activation function | MAE | RMSE | WMAPE |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| STGCN (Cheb) | STGCN_ChebConv (Ks=3, Kt=3) | sym | GLU | 6.047641 | 11.888628 | 11.909882% |
| STGCN (1<sup>st</sup>) | STGCN_GCNConv (Kt=3) | sym | GLU | 5.997484 | 11.498759 | 11.811108% |

### PEMS-BAY (15/30/60 mins) (train: val: test = 70: 15: 15)
#### 15 mins (H=3)
| Model (paper) | Model (code) | Laplacian matrix type | Gated activation function | MAE | RMSE | WMAPE |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| STGCN (Cheb) | STGCN_ChebConv (Ks=3, Kt=3) | sym | GLU | 1.504175 | 3.031081 | 2.420486% |
| STGCN (1<sup>st</sup>) | STGCN_GCNConv (Kt=3) | sym | GLU | 1.472308 | 2.987471 | 2.369206% |

#### 30 mins (H=6)
| Model (paper) | Model (code) | Laplacian matrix type | Gated activation function | MAE | RMSE | WMAPE |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| STGCN (Cheb) | STGCN_ChebConv (Ks=3, Kt=3) | sym | GLU | 1.919455 | 3.964940 | 3.088833% |
| STGCN (1<sup>st</sup>) | STGCN_GCNConv (Kt=3) | sym | GLU | 1.910708 | 3.948517 | 3.074757% |

#### 60 mins (H=12)
| Model (paper) | Model (code) | Laplacian matrix type | Gated activation function | MAE | RMSE | WMAPE |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| STGCN (Cheb) | STGCN_ChebConv (Ks=3, Kt=3) | sym | GLU | 2.308847 | 4.690512 | 3.715672% |
| STGCN (1<sup>st</sup>) | STGCN_GCNConv (Kt=3) | sym | GLU | 2.306092 | 4.701984 | 3.711238% |

# STGCN
The PyTorch version of STGCN implemented by the paper *Spatio-Temporal Graph Convolutional Networks:
A Deep Learning Framework for Traffic Forecasting*

## Paper
https://arxiv.org/abs/1709.04875

## Model structure
<img src="./figure/stgcn_model_structure.png" style="zoom:100%" />

## Other implementations
1. https://github.com/Davidham3/STGCN (MXNet)
2. https://github.com/VeritasYin/STGCN_IJCAI-18 (TensorFlow v1)

## Differents of code between mine and author's
1. Fix bugs 
2. Add Early Stopping approach
3. Add Dropout approach
4. Offer a better set of hyperparameters

## The result for road traffic prediction on dataset PeMSD7(M)(15 mins)
|  Model(paper)  |  Model(code)  |  Gated activation function  |  Laplacian matrix type  |  MAE  |  MAPE  |  RMSE  |
|  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |
|  STGCN(Cheb)  |  STGCN_ChebConv(Ks=3)  |  GLU  |  L_sym  |  2.219458  |  5.137035%  |  3.966818  |
|  STGCN(1<sup>st</sup>)  |  STGCN_GCNConv  |  GLU  |  L_sym  |  2.162756  |  5.018773%  |  3.910300  |

## The result for road traffic prediction on dataset PeMSD7(M)(30 mins)
|  Model(paper)  |  Model(code)  |  Gated activation function  |  Laplacian matrix type  |  MAE  |  MAPE  |  RMSE  |
|  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |
|  STGCN(Cheb)  |  STGCN_ChebConv(Ks=3)  |  GLU  |  L_sym  |  2.959860  |  7.226268%  |  5.334936  |
|  STGCN(1<sup>st</sup>)  |  STGCN_GCNConv  |  GLU  |  L_sym  |  2.816108  |  6.833110%  |  5.181678  |

## The result for road traffic prediction on dataset PeMSD7(M)(45 mins)
|  Model(paper)  |  Model(code)  |  Gated activation function  |  Laplacian matrix type  |  MAE  |  MAPE  |  RMSE  |
|  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |
|  STGCN(Cheb)  |  STGCN_ChebConv(Ks=3)  |  GLU  |  L_sym  |  3.388877  |  8.455504%  |  6.092015  |
|  STGCN(1<sup>st</sup>)  |  STGCN_GCNConv  |  GLU  |  L_sym  |  3.201253  |  7.947086%  |  5.842915  |

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

## Applications
According to the paper, STGCN could be applied into general spatio-temporal structured sequence forecasting scenarios. We find it could be used for PM2.5 prediction.

## Differents of code between mine and author's
1. I fix the bug of calculating the normalized laplacian matrix. In the author's code, it calculated as I_n + \widetildeD^{-1/2} * \widetildeW * \widetildeD^{-1/2} which is wrong, obviously. In my code, it calculated as \widetildeD^{-1/2} * \widetildeW * \widetildeD^{-1/2} according to the paper *Semi-Supervised Classification with Graph Convolutional Networks*.
2. I add the early stopping approach.
3. I enable the dropout approach for training and testing.
4. I offer a better set of hyperparameters rather than the author's code offered.
5. We find STGCN could be used for PM 2.5 prediction, so I add main function to achieve it.

## The result for road traffic prediction on dataset PeMSD7(M)(45 mins)
|  Model(paper)  |  Model(code)  |  MAE  |  MAPE  |  RMSE  |
|  ----  |  ----  |  ----  |  ----  |  ----  |
|  STGCN(Cheb)  |  STGCN_GC_CPA  |  3.163535  |  7.8568%  |  5.878053  |
|  STGCN(1<sup>st</sup>)  |  STGCN_GC_LWL  |  3.099546  |  7.7346%  |  5.731380  |

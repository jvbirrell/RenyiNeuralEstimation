# RenyiNeuralEstimation
Neural-network estimation of the Renyi divergence, R_alpha, between two n-dimensional Gaussians via 2 different methods
1) Renyi-Donsker-Varadhan (DV) method from Variational Representations and Neural Network Estimation for Renyi Divergences,  SIAM Journal on Mathematics of Data Science, 2021
2) infimal-convolution method from Function-space regularized Renyi divergences, ICLR 2023


Sample usage:

alpha=2
run_num=1
dim=1000
Lrate=0.001
method='inf_conv_Renyi'
IC_activation='polysoftplus'
rescaled=0
mu=0.5

python3 Renyi_variance_test.py  $alpha $run_num $dim $Lrate $method $IC_activation $rescaled $mu


Input arguments:

1) The value of alpha
2) The current run number (so results from multiple runs are saved in separate files)
3) The dimension of the Gaussians
4) Learning rate (Adam optimizer)
5) The method name (DV_Renyi or inf_conv_Renyi)
6) The final activation layer for the inf_conv_Renyi method (Currently implemented options are abs, elu, relu, polysoftplus but the user can add their own, with the requirement is that the output must be <=0. This argument is not used by the DV_Renyi method.) 
7) Set to 1 to use rescaled Renyi (i.e., alpha*R_alpha)
8) Separation between the means of the Gaussians


Notes: 
1) Requires Tensorflow 1.15
2) The discriminator is implemented as a fully-connected neural network.  This architecture can be modified by the user to implement more general Gamma-Renyi divergences from Function-space regularized Renyi divergences, ICLR 2023.  
3) Both distributions have covariance=I but these can be changed by the user. The code makes no assumptions regarding the covariance matrices.


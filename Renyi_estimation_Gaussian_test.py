#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Author: Jeremiah Birrell

#Neural estimation of the Renyi divergences between two n-dimensional Gaussians via 2 different methods
#1) DV method from Variational Representations and Neural Network Estimation for Renyi Divergences,  SIAM Journal on Mathematics of Data Science}, 2021
#2) CC method from Function-space regularized Renyi divergences, ICLR 2023

import sys
import tensorflow as tf
import numpy as np
import csv
import os
alpha_str =sys.argv[1]
alpha=float(alpha_str)  # alpha>1
run_num=int(sys.argv[2])    # run number
n=int(sys.argv[3]) #dimension of gaussians
Lrate=float(sys.argv[4])    # learning rate
method=sys.argv[5] #DV_Renyi,inf_conv_Renyi
IC_activation=sys.argv[6] #abs, elu, relu, polysoftplus
rescaled=int(sys.argv[7]) #1 if use rescaled Renyi
mu_str=sys.argv[8]
mu=float(mu_str) #mean of Q distriubtion


epochs = 10000
mb_size = 500
#number of nodes in each hidden layer (can have more than one hidden layer)
hidden_layers = [64] 



#save estimate every SF iterations
SF = 100
#samples for estimating Df
N=50000





#means, and covariances of the Gaussian r.v.s
mu_p=np.zeros((n,1))
mu_q=np.zeros((n,1))
mu_q[0]=mu



Sigma_p=np.identity(n)
Sigma_q=np.identity(n)



#M@np.transpose(M)=Sigma
Mp=np.linalg.cholesky(Sigma_p)
Mq=np.linalg.cholesky(Sigma_q)
    


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1.0 / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


#construct variables for the neural networks
def initialize_W(layers):
    W_init=[]
    num_layers = len(layers)
    for l in range(0,num_layers-1):
        W_init.append(xavier_init(size=[layers[l], layers[l+1]]))
    return W_init

def initialize_NN(layers,W_init):
    NN_W = []
    NN_b = []
    num_layers = len(layers)
    for l in range(0,num_layers-1):
        W = tf.Variable(W_init[l])
        b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
        NN_W.append(W)
        NN_b.append(b)
    return NN_W, NN_b



        
Sigma_alpha=alpha*Sigma_q+(1.0-alpha)*Sigma_p       
       
alpha_scaling=1.
if rescaled==1:
    alpha_scaling=alpha


#Exact R_alpha(P||Q)
Renyi_exact=1.0/2.0*np.matmul(np.transpose(mu_q-mu_p),np.matmul(np.linalg.inv(Sigma_alpha),mu_q-mu_p))-1.0/(2.0*alpha*(alpha-1.0))*np.math.log(np.linalg.det(Sigma_alpha)/(np.math.pow(np.linalg.det(Sigma_p),1.0-alpha)*np.math.pow(np.linalg.det(Sigma_q),alpha)))
divergence_exact=alpha_scaling*Renyi_exact[0][0]


#variable for Q
X = tf.placeholder(tf.float32, shape=[None, n])
#variable for P
Z = tf.placeholder(tf.float32, shape=[None, n])




layers=[n]+hidden_layers+[1]
    

W_init=initialize_W(layers)
D_W, D_b = initialize_NN(layers,W_init)
theta_D = [D_W, D_b]

  
def discriminator(x):
    num_layers = len(D_W) + 1
    
    h = x
    for l in range(0,num_layers-2):
        W = D_W[l]
        b = D_b[l]
        h = tf.nn.relu(tf.add(tf.matmul(h, W), b))
    
    W = D_W[-1]
    b = D_b[-1]
    out =  tf.matmul(h, W) + b

    
    return out        
 

#IC_activation: abs, elu, relu, polysoftplus

if IC_activation=='abs':
    def IC_final_layer(y):
        return -tf.math.abs(y)
elif IC_activation=='elu':
    def IC_final_layer(y):
        return -(tf.nn.elu(y)+1.)
elif IC_activation=='relu':
    def IC_final_layer(y):
        return -tf.nn.relu(y)
elif IC_activation=='polysoftplus':
    def IC_final_layer(y):
        return -(1.+(1./(1.+tf.nn.relu(-y))-1)*(1.-tf.sign(y))/2. +y*(tf.sign(y)+1.)/2. )

def sample_P(N_samp):
    return np.transpose((mu_p+np.matmul(Mp,np.random.normal(0., 1.0, size=[n, N_samp]))))

def sample_Q(N_samp):
    return np.transpose((mu_q+np.matmul(Mq,np.random.normal(0., 1.0, size=[n, N_samp]))))

P_data=discriminator(Z)
Q_data=discriminator(X)

P_max=tf.reduce_max((alpha-1.0)*P_data)
Q_max=tf.reduce_max(alpha*Q_data)

if method=='DV_Renyi':
    objective=alpha_scaling/(alpha-1.0)*tf.math.log(tf.reduce_mean(tf.math.exp(((alpha-1.0)*P_data-P_max)/alpha_scaling)))+1.0/(alpha-1.0)*P_max-1.0/alpha*Q_max-alpha_scaling/alpha*tf.math.log(tf.reduce_mean(tf.math.exp((alpha*Q_data-Q_max)/alpha_scaling)))
elif method=='inf_conv_Renyi':
    objective=tf.reduce_mean(IC_final_layer(Q_data))+alpha_scaling/(alpha-1.)*tf.math.log(tf.reduce_mean(tf.math.pow(-IC_final_layer(P_data)/alpha_scaling,(alpha-1.)/alpha)))+(alpha_scaling/alpha)*(np.math.log(alpha)+1.)



#AdamOptimizer
solver = tf.train.AdamOptimizer(learning_rate=Lrate).minimize(-objective, var_list=theta_D)



config = tf.ConfigProto(device_count={'GPU': 0})
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())



divergence_array=np.zeros(epochs//SF+1)




#use N samples for estimation of Df
Q_plot_samples=sample_Q(N)
P_plot_samples=sample_P(N)


j=0
for it in range(epochs+1):
       
    if it>0:

            X_samples=sample_Q(mb_size)
            Z_samples=sample_P(mb_size) 

            sess.run(solver, feed_dict={X: X_samples, Z: Z_samples})
    
    if it % SF == 0:                
       
        
        X_samples=Q_plot_samples
        Z_samples=P_plot_samples
        
        
        divergence_array[j]=sess.run( objective, feed_dict={X: X_samples, Z: Z_samples})
        
        print()
        print(method)
        if rescaled==1 and not(method=='alpha_div'):
            print('rescaled')
        if method=='inf_conv_Renyi':
            print(IC_activation)
        print('mu: {}'.format(mu))
        print('alpha: {}'.format(alpha))
        print('Iter: {}'.format(it))
        print('Divergence: {:.6f}'.format(divergence_exact))
        print('Divergence est.: {:.6f}'.format(divergence_array[j]))
        print('Error: {:.6f}'.format(divergence_array[j]-divergence_exact))
        print('Rel. err.: {:.6f}'.format(1.0-divergence_array[j]/divergence_exact))
        
        j=j+1
        
    
        
        
rel_err=1.-divergence_array/divergence_exact

layers_str=''
for layer_dim in hidden_layers:
    layers_str=layers_str+' '+str(layer_dim)
print()
print('Hidden Layers:'+layers_str)

test_name=method+'_Gaussian_est/'
if method=='inf_conv_Renyi':
    test_name=IC_activation+'_'+test_name
if rescaled==1:
    test_name='rescaled_'+test_name

if not os.path.exists(test_name):
    os.makedirs(test_name)
    
with open(test_name+'Exact_divergence_alpha_'+alpha_str+'_mu_'+mu_str+'_Lrate_{:.1e}'.format(Lrate)+'_epochs_'+str(epochs)+'_mbsize_'+str(mb_size)+'_dim_'+str(n)+'_layers_'+layers_str+'run'+str(run_num)+'.csv', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow([divergence_exact])  

with open(test_name+'Divergence_est_alpha_'+alpha_str+'_mu_'+mu_str+'_Lrate_{:.1e}'.format(Lrate)+'_epochs_'+str(epochs)+'_mbsize_'+str(mb_size)+'_dim_'+str(n)+'_layers_'+layers_str+'run'+str(run_num)+'.csv', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(divergence_array)  

with open(test_name+'rel_err_alpha_'+alpha_str+'_mu_'+mu_str+'_Lrate_{:.1e}'.format(Lrate)+'_epochs_'+str(epochs)+'_mbsize_'+str(mb_size)+'_dim_'+str(n)+'_layers_'+layers_str+'run'+str(run_num)+'.csv', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(rel_err)  

  

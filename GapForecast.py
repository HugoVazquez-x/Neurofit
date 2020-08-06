# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 13:32:55 2020

@author: hugov
"""


import math
import numpy as np
from neurofit_rl import neurofit
import os
import sys
from MLNN import MLNN
from tensorflow import keras

Nvar=2  # Z,A
Npar2=0 #parameter of correction
Nres=1  #the result

#Read input data
dataset = np.loadtxt(fname = "HalfLifeForecast.dat")
error = dataset[:,dataset.shape[1]-1]
dataset = dataset[:,:dataset.shape[1]-1]

#Architecture parameters
param_code = str(sys.argv[1])
nb_hid_lay = int(sys.argv[2])
arch = []
for i in range(3,len(sys.argv)):
    arch.append(int(sys.argv[i]))

#Split the dataset  into K-fold validation (data_train and data_eval)
k = 4
nb_data_eval = math.ceil(dataset.shape[0]*(1/k))
nb_data_train = dataset.shape[0] - nb_data_eval

validation_scores = []

def additional_param(param_list):
    """
    This function will take param in the same order than in database and compute the additional
    parameter discribe by user.
    The user have to give the number of additional parameter desired and give the functions expected
    in return

    Input : List of parameters (do not change)
    Output <class 'list'>: Number of additional parmaters,additionnal parameters
    """
    N_add_param = 0

    return [N_add_param]


for fold in range(k):
    print("----------K-FOLD %2d-----------------------------" % fold)
    database_eval = dataset[nb_data_eval * fold: nb_data_eval * (fold + 1),:]
    database = np.concatenate((dataset[:nb_data_eval * fold,:],
                              dataset[nb_data_eval * (fold + 1):,:]),axis=0)
    
    database[:,database.shape[1]-2:] = np.log10(database[:,database.shape[1]-2:])
    database_eval[:,database_eval.shape[1]-2:] = np.log10(database_eval[:,database_eval.shape[1]-2:])



    #Decrypte param_code
    if(param_code == "1000"):
        param = database[:,2].reshape(database[:,2].shape[0],1)
        param_eval = database_eval[:,2].reshape(database_eval[:,2].shape[0],1)
        Npar1=1  #bet2
    elif (param_code == "0100"):
        param = database[:,3].reshape(database[:,3].shape[0],1)
        param_eval = database_eval[:,3].reshape(database_eval[:,3].shape[0],1)
        Npar1=1  #Sn
    elif (param_code == "0010") : 
        param = database[:,4].reshape(database[:,4].shape[0],1)
        param_eval = database_eval[:,4].reshape(database_eval[:,4].shape[0],1)
        Npar1=1  #Sp
    elif ( param_code == "0001") :
        param = database[:,5].reshape(database[:,5].shape[0],1)
        param_eval = database_eval[:,5].reshape(database_eval[:,5].shape[0],1)
        Npar1=1  #Qbet
    elif (param_code == "1100" ):
        param = database[:,2:4]
        param_eval = database_eval[:,2:4]
        Npar1=2  #bet2,Sn
    elif (param_code == "1010" ):
        param = np.concatenate((database[:,2].reshape(database[:,2].shape[0],1),database[:,4].reshape(database[:,4].shape[0],1)),axis=1)
        param_eval = np.concatenate((database_eval[:,2].reshape(database_eval[:,2].shape[0],1),database_eval[:,4].reshape(database_eval[:,4].shape[0],1)),axis=1)
        Npar1=2  #bet2,Sp
    elif (param_code == "1001") :
        param = np.concatenate((database[:,2].reshape(database[:,2].shape[0],1),database[:,5].reshape(database[:,4].shape[0],1)),axis=1)
        param_eval = np.concatenate((database_eval[:,2].reshape(database_eval[:,2].shape[0],1),database_eval[:,5].reshape(database_eval[:,4].shape[0],1)),axis=1)
        Npar1=2  #bet2,Qbet
    elif ( param_code == "0110"):
        param = database[:,3:5]
        param_eval = database_eval[:,3:5]
        Npar1=2  #Sn,Sp
    elif (param_code == "0101"):
        param = np.concatenate((database[:,3].reshape(database[:,3].shape[0],1),database[:,5].reshape(database[:,4].shape[0],1)),axis=1)
        param_eval = np.concatenate((database_eval[:,3].reshape(database_eval[:,3].shape[0],1),database_eval[:,5].reshape(database_eval[:,4].shape[0],1)),axis=1)
        Npar1=2  #Sn,Qbet
    elif ( param_code == "0011"):
        param = database[:,4:6]
        param_eval = database_eval[:,4:6]
        Npar1=2  #Sp,Qbet
    elif ( param_code == "1110"):
        param = database[:,2:5]
        param_eval = database[:,2:5]
        Npar1=3  #bet2,Sn,Sp
    elif ( param_code == "1101"):
        param = np.concatenate((database[:,2:4],database[:,5].reshape(database[:,5].shape[0],1)),axis=1)
        param_eval = np.concatenate((database_eval[:,2:4],database_eval[:,5].reshape(database_eval[:,5].shape[0],1)),axis=1)
        Npar1=3  #bet2,Sn,Qbet
    elif ( param_code == "0111"):
        param = database[:,3:6]
        param_eval = database_eval[:,3:6]
        Npar1=3  #Sn,Sp,Qbet
    elif (param_code == "1011"):
        param = np.concatenate((database[:,2].reshape(database[:,2].shape[0],1),database[:,4:6]),axis=1)
        param_eval = np.concatenate((database_eval[:,2].reshape(database_eval[:,2].shape[0],1),database_eval[:,4:6]),axis=1)
        Npar1=3  #bet2,Sp,Qbet
    elif (param_code == "1111"):
        param = database[:,2:6]
        param_eval = database_eval[:,2:6]
        Npar1=4  #bet2,Sn,Sp,Qbet


    #choose parameters to help the network
    database = np.concatenate((database[:,:2],param,database[:,6:8]), axis=1 )
    database_eval = np.concatenate((database_eval[:,:2],param_eval,database_eval[:,6:8]),axis=1 )



    nnet = MLNN(database, database_eval, Nvar, Npar1, Npar2, Nres, [0,0], [0,0],additional_param,[0,0],1)
    
    #Parameters of the Neural Networks
    num_hidden_layers = nb_hid_lay
    architecture = arch
    act_func = 'relu'
    successive_fit_numb = 5
    epoch = [2000,2000,2000,2000,2000]
    batch = [10,10,10,10,10]
    
    #Force Keras to work with 'float64'
    keras.backend.set_floatx('float64')
    
    #Building the model
    #if os.path.isfile('last_model.h5'):
    #    nnet.model = keras.models.load_model('last_model.h5')
    #else:
    nnet.buildModel(num_hidden_layers,architecture,act_func)
      
    
    #Training the model
    nnet.trainModel(successive_fit_numb,epoch,batch)
    
    #Plot training resume
    #nnet.plotTrainningHistory(successive_fit_numb)
    
    #Performances
    validation_score = nnet.model.evaluate(nnet.x_eval,nnet.y_eval,verbose =0)
    print(nnet.model.metrics_names)
    print(validation_score)
    validation_scores.append(math.sqrt(validation_score[0]))
    
    #Plot performances
    key = list(nnet.trainModel_history[0].history.keys())
    val_mae = key[3]
    print("val_mae :" ,nnet.trainModel_history[-1].history[val_mae])
    
    os.remove("last_model.h5")
    os.remove("weights.best.hdf5")
    os.remove("mean.txt")
    os.remove("std.txt")

print(validation_scores)
validation_score = np.average(validation_scores)









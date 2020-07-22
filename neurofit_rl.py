# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 11:40:46 2020

@author: hugov
"""
from tensorflow import keras
from MLNN import MLNN
import numpy as np
import os
from tools import x_in_y


def neurofit(database, database_eval, Nvar, Npar1, Npar2, Nres, bornes, list_pts, additional_param,nb_hid_lay,arch,Nstep,istep,exp_values):

    emul = MLNN(database, database_eval, Nvar, Npar1, Npar2, Nres, bornes, list_pts,additional_param,exp_values,istep)

    #Parameters of the Neural Networks
    num_hidden_layers = nb_hid_lay
    architecture = arch
    act_func = 'relu'
    successive_fit_numb = 1
#    #The number of epoch decrease in 1/x with regards to istep progress
#    epoch_max = 1000
#    epoch_min = 300
#    a = (Nstep/(Nstep-1))*(epoch_max-epoch_min)
#    b = (1/(Nstep-1))*(Nstep*epoch_min-epoch_max)
#    epoch = [math.ceil(a/(istep+1) + b)]
#    #The number of batch increase in 1/x with regars to istep progress
#    batch_max = 32
#    batch_min = 5
#    a1 = (Nstep/(1-Nstep))*(batch_min-batch_max)
#    b1 = batch_min + a1
#    batch = [math.ceil( -a1/(istep+1) + b1)]
    epoch = [2000]
    batch = [10]

    #Force Keras to work with 'float64'
    keras.backend.set_floatx('float64')

    #Building the model
    if os.path.isfile('last_model.h5'):
        emul.model = keras.models.load_model('last_model.h5')
    else:
        emul.buildModel(num_hidden_layers,architecture,act_func)
  
    
    #Training the model
    emul.trainModel(successive_fit_numb,epoch,batch)

    #Plot training resume
    emul.plotTrainningHistory(successive_fit_numb)
    
    #Complete param_rms.txt for further use
    emul.updateParamRmsFile()  

    #Plot estimate function
    emul.plotModel()

    #Minimization
    Pred = []

    ##One minimization from best prediction
    pred = emul.minimizing(mode = 'from_best')
    Pred.append(pred)

    ##Nine minizations from random initial guess
    for i in range(9):
        pred = emul.minimizing(mode = 'random')
        if (x_in_y(pred,Pred) == False):
            Pred.append(pred)
    Pred = np.array(Pred)

    #Writing the results on two different files
    if(istep == Nstep-1) :
        emul.writeResults("one_ligne_results.txt",Pred)

    emul.writeResults("all_results.txt",Pred)
    
    #print("Predicted Rms for gam0 = 1 and fcorr = 1 : " ,emul.getRms(np.array([1,1,1])))
    
    return Pred

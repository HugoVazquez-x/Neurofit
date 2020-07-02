# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 11:40:46 2020

@author: hugov
"""
from tensorflow import keras
from MLNN import MLNN
import numpy as np
import os

def x_in_y(x, y):
    # check if x is a nested list
    if any(isinstance(i, list) for i in x):
        return all((any((set(x_).issubset(y_) for y_ in y)) for x_ in x))
    else:
        return any((set(x).issubset(y_) for y_ in y))

def neurofit(database, database_eval, Nvar, Npar1, Npar2, Nres, bornes, list_pts, additional_param,nb_hid_lay,arch):

    emul = MLNN(database, database_eval, Nvar, Npar1, Npar2, Nres, bornes, list_pts,additional_param)

    #Parameters of the Neural Networks
    num_hidden_layers = nb_hid_lay
    architecture = arch
    act_func = 'relu'
    successive_fit_numb = 1
    epoch = [1000]
    batch = [10]

    #Force Keras to work with 'float64'
    keras.backend.set_floatx('float64')

    #Building the model
    if os.path.isfile('last_model.h5'):
        emul.model = keras.models.load_model('last_model.h5')
    else:
        emul.build_model(num_hidden_layers,architecture,act_func)

    #Training the model
    emul.train_model(successive_fit_numb,epoch,batch)

    #Plot results
    #emul.training_view(successive_fit_numb)


    #Minimization
    Pred_results = []
    for i in range(10):
        pred = emul.minimizing()
        if (x_in_y(pred,Pred_results) == False):
            Pred_results.append(pred)
    Pred_results = np.array(Pred_results)
    #Writing the results
    emul.writing_results(Pred_results)

    return Pred_results

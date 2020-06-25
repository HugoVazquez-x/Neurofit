# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 11:40:46 2020

@author: hugov
"""
import matplotlib.pyplot as plt
from MLNN import MLNN

def neurofit(database, database_eval, Nvar, Npar1, Npar2, Nres, bornes, list_pts):
    
    emul = MLNN(database, database_eval, Nvar, Npar1, Npar2, Nres, bornes, list_pts)    
    
    #Parameters of the Neural Networks
    num_hidden_layers = 1
    architecture = [3,10]
    act_func = 'relu'
    successive_fit_numb = 1
    epoch = [500]
    batch = [10]
    
    #Training the model
    emul.build_model(num_hidden_layers,architecture,act_func)
    emul.train_model(successive_fit_numb,epoch,batch)
    emul.training_view(successive_fit_numb)
    
    #weights of model
    #print("waeight of model : ", emul.model.get_weights())
    #Minimization
    pred = emul.minimizing()
    
#    plt.figure(figsize=(10,10))
#    plt.plot(emul.a,emul.rms)
#    plt.title("Rms function of a " )
#    plt.ylabel('rms')
#    plt.xlabel('a')
    
    return pred
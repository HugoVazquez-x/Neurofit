# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:46:39 2020

@author: hugov
"""
import math
import numpy as np
from scipy.optimize import least_squares

class Minimize():
    def __init__(self,Nvar, Npar1, Npar2, Nres, bornes, list_pts,mean,std):
        
        self.Nvar = Nvar
        self.Npar1 = Npar1
        self.Npar2 = Npar2
        self.Nres = Nres
        self.bornes = bornes
        
        self.list_pts = np.array(list_pts)
        self.list_pts = self.list_pts.reshape((self.list_pts.shape[0],1))
        
        self.mean = mean
        self.std = std
        
    def f_to_minimize(self,param):
        
        
        param = param.reshape((1,param.shape[0]))
        test_data = np.empty((self.list_pts.shape[0], self.Nvar + self.Npar1 + self.Npar2))
        test_data[:,:self.Nvar] = self.list_pts
        test_data[:,self.Nvar:self.Nvar + self.Npar1] = param 
        
        #Ajout de paramètre suplémentaire ici a*x
        more_param = test_data[:,0]*test_data[:,1]
        more_param = more_param.reshape((more_param.shape[0],1))
        test_data[:,self.Nvar + self.Npar1:] = more_param
        
        #Normalisation des données
        test_data -= self.mean[:self.Npar1 + self.Npar2 + self.Nvar]
        test_data /= self.std[:self.Npar1 + self.Npar2 + self.Nvar]
        
        #Prediction of the model
        pred  = self.model_prediction_test(test_data)  
        pred = pred.reshape(pred.shape[0],)

        
        return pred
        
    
    def minimizing(self):
        
        #Initial guess on parameters.
        param_0 = np.empty((self.Npar1))
        for i in range(self.Npar1):
            param_0[i] = (self.bornes[i][1]- self.bornes[i][0])*np.random.random_sample() + self.bornes[i][0]   # random number in the bornes
        
        
        #Reshape of bound list to fit with least_square
        bound_reshape = [[],[]]
        for i in range(len(self.bornes)):
             bound_reshape[0].append(self.bornes[i][0])
             bound_reshape[1].append(self.bornes[i][1])
        
        print("avant :", param_0)
        #minimisation
        res = least_squares(self.f_to_minimize,param_0, bounds = bound_reshape)
        print("aprés :", res.x)
       
        return res.x
        
        
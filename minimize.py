# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:46:39 2020

@author: hugov
"""

import numpy as np
from scipy.optimize import least_squares
import math

class Minimize():
    def __init__(self,Nvar, Npar1, Npar2, Nres, bornes, list_pts,mean,std,additional_param):

        self.Nvar = Nvar
        self.Npar1 = Npar1
        self.Npar2 = Npar2
        self.Nres = Nres
        self.bornes = bornes

        self.list_pts = np.array(list_pts)
        self.list_pts = self.list_pts.reshape((self.list_pts.shape[0],1))

        self.mean = mean
        self.std = std

        self.additional_param = additional_param

    def f_to_minimize(self,param):


        param = param.reshape((1,param.shape[0]))
        test_data = np.empty((self.list_pts.shape[0], self.Nvar + self.Npar1 ))
        test_data[:,:self.Nvar] = self.list_pts
        test_data[:,self.Nvar:self.Nvar + self.Npar1] = param

        #Addition of addtional parameters
        test_data = self.complete_dataset(test_data, self.Nvar,self.Npar1,self.additional_param)

        #Normalisation of data
        test_data -= self.mean[:len(self.mean)-self.Nres]
        test_data /= self.std[:len(self.std)-self.Nres]

        #Prediction of the model

        pred  = self.model_prediction_test(test_data)
        pred = pred.reshape(pred.shape[0],)

        return pred


    def minimizing(self,mode):

        #Reshape of bound list to fit with least_square
        bound_reshape = [[],[]]
        for i in range(len(self.bornes)):
             bound_reshape[0].append(self.bornes[i][0])
             bound_reshape[1].append(self.bornes[i][1])
        
        if (mode == 'random'):
            #Compute the average rms with parameters in data_train
            rms_mean = 0
            for i in range(self.parameters.shape[0]):
                rms_mean += self.rms(self.parameters[i])
            
            rms_mean /= self.parameters.shape[0]
            rms_pred = rms_mean + 10
            
            iteration = 0
            #minimisation
            while (rms_pred > rms_mean):
                #Break conditions
                if(iteration == 20):
                    break  
                #Initial guess on parameters
                param_0 = np.empty((self.Npar1))
                for i in range(self.Npar1):
                    param_0[i] = (self.bornes[i][1]- self.bornes[i][0])*np.random.random_sample() + self.bornes[i][0]   # random number in the bornes
                
                #We compute severals gradient descent with differents algorithms methods to find the best minimum
                res_trf = least_squares(self.f_to_minimize,param_0, bounds = bound_reshape, method = 'trf')
                res_dogbox = least_squares(self.f_to_minimize,param_0, bounds = bound_reshape, method = 'dogbox')
                res_lm = least_squares(self.f_to_minimize,param_0, method = 'lm')
                print("--------Minimisation Iteration %2d----------" % iteration)
                for i in range(2):
                   #We use the last guess for input parameters 
                   res_trf = least_squares(self.f_to_minimize,res_dogbox.x, bounds = bound_reshape, method = 'trf')
                   res_dogbox = least_squares(self.f_to_minimize,res_trf.x, bounds = bound_reshape, method = 'dogbox')
                #We find which algorithm method find the best prediction and compute its rms 
                list_res = [res_trf,res_dogbox,res_lm]
                rms_trf = self.rms(res_trf.x)
                rms_dogbox = self.rms(res_dogbox.x)
                rms_lm = self.rms(res_lm.x)
                list_rms = [rms_trf,rms_dogbox,rms_lm]
                rms_pred = min(list_rms)
                rms_best_index = list_rms.index(rms_pred)
                res = list_res[rms_best_index]
                iteration += 1
            
        if (mode == 'from_best'):            
            best_param_index = 0
            rms_min = 100
            for i in range(self.parameters.shape[0]):
                r = self.rms(self.parameters[i])
                if(r < rms_min):
                    rms_min = r
                    best_param_index = i
            
            param_0 = self.parameters[best_param_index]
            #We compute severals gradient descent with differents algorithms methods to find the best minimum
            res_trf = least_squares(self.f_to_minimize,param_0, bounds = bound_reshape, method = 'trf')
            res_dogbox = least_squares(self.f_to_minimize,param_0, bounds = bound_reshape, method = 'dogbox')
            res_lm = least_squares(self.f_to_minimize,param_0, method = 'lm')
              
            for i in range(2):
                #We mix minimisation algorithme from last result 
                res_trf = least_squares(self.f_to_minimize,res_dogbox.x, bounds = bound_reshape, method = 'trf')
                res_dogbox = least_squares(self.f_to_minimize,res_trf.x, bounds = bound_reshape, method = 'dogbox')
                   
                print("res_trf = ",res_trf.x ," res_dogbox = ",res_dogbox.x)
            #We find wich algorithm method find the best prediction and compute its rms 
            list_res = [res_trf,res_dogbox,res_lm]
            rms_trf = self.rms(res_trf.x)
            rms_dogbox = self.rms(res_dogbox.x)
            rms_lm = self.rms(res_lm.x)
            list_rms = [rms_trf,rms_dogbox,rms_lm]
            rms_pred = min(list_rms)
            print("rms pred from best = ",rms_pred)
            rms_best_index = list_rms.index(rms_pred)
            res = list_res[rms_best_index]            
            
        return res.x
    
    def rms(self,param):
        val = []
        for j in range(self.exp_values.shape[0]):
            val.append(self.fct( self.exp_values[j,0], param[0] ,param[1],param[2]))
        return math.sqrt(sum((self.exp_values[:,1]-np.array(val))**2))
   
    def fct(self,x,a,b,c):
        return c*math.cos(a*x+b)
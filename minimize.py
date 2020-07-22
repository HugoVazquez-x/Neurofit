# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:46:39 2020

@author: hugov
"""

import numpy as np
from scipy.optimize import least_squares
import math
import os

class Minimize():
    def __init__(self,Nvar, Npar1, Npar2, Nres, bornes, list_pts,mean,std,additional_param):

        self.Nvar = Nvar
        self.Npar1 = Npar1
        self.Npar2 = Npar2
        self.Nres = Nres
        self.bornes = bornes

        self.list_pts = list_pts

        self.mean = mean
        self.std = std

        self.additional_param = additional_param

    def f_to_minimize(self,param):
        
        #Get a dataset of list_pts with "param" used as parameters 
        data = self.getDataset(param)
        
        #Predictions
        y_pred = self.getPrediction(data)
        
        return y_pred


    def minimizing(self,mode):

        #Reshape of bound list to fit with least_square
        bound_reshape = [[],[]]
        for i in range(len(self.bornes)):
             bound_reshape[0].append(self.bornes[i][0])
             bound_reshape[1].append(self.bornes[i][1])
        
        if (mode == 'random'):
            #Initial guess on parameters
            param_0 = np.empty((self.Npar1))
            for i in range(self.Npar1):
                param_0[i] = (self.bornes[i][1]- self.bornes[i][0])*np.random.random_sample() + self.bornes[i][0]   # random number in the bornes
            
            #We compute severals gradient descent with differents algorithms methods to find the best minimum
            res_trf = least_squares(self.f_to_minimize,param_0, bounds = bound_reshape, method = 'trf')
            res_dogbox = least_squares(self.f_to_minimize,param_0, bounds = bound_reshape, method = 'dogbox')
            
            #We find which algorithm method find the best prediction and compute its rms 
            list_res = [res_trf,res_dogbox]
            rms_trf = self.getRms(res_trf.x)
            rms_dogbox = self.getRms(res_dogbox.x)
            list_rms = [rms_trf,rms_dogbox]
            rms_pred = min(list_rms)
            rms_best_index = list_rms.index(rms_pred)
            res = list_res[rms_best_index]
            output = res.x
            output = np.append(output,[self.getRms(res.x)], axis = 0)
            
            return output
            
        if (mode == 'from_best'):            

            param_0 = self.getBestParam()
            print("rms_min = ",self.getRms(param_0),"param_0 =",param_0)
            
            #We compute severals gradient descent with differents algorithms methods to find the best minimum
            res_trf = least_squares(self.f_to_minimize,param_0, bounds = bound_reshape, method = 'trf')
            res_dogbox = least_squares(self.f_to_minimize,param_0, bounds = bound_reshape, method = 'dogbox')
            
            #We find wich algorithm method find the best prediction and compute its rms 
            list_res = [res_trf,res_dogbox]
            rms_trf = self.getRms(res_trf.x)
            rms_dogbox = self.getRms(res_dogbox.x)
            list_rms = [rms_trf,rms_dogbox]
            rms_pred = min(list_rms)
            rms_best_index = list_rms.index(rms_pred)
            res = list_res[rms_best_index]            
            output = res.x
            output = np.append(output,[self.getRms(res.x)], axis = 0)
            
            return output
            
        
    
    def getRms(self,param):
        val = self.f_to_minimize(param)
        return math.sqrt(sum((val)**2)/val.shape[0])
   
        
    def getDataset(self,param):
        param = param.reshape((1,param.shape[0]))
        data = np.empty((self.list_pts.shape[0], self.Nvar + self.Npar1 + self.Npar2 + self.Nres ))
        data[:,:self.Nvar] = self.list_pts
        data[:,self.Nvar:self.Nvar + self.Npar1] = param

        #Addition of addtional parameters
        data = self.completeDataset(data, self.Nvar,self.Npar1,self.Npar2,self.Nres,self.additional_param)

        #Normalisation of data
        data -= self.mean[:len(self.mean)-self.Nres]
        data /= self.std[:len(self.std)-self.Nres]
        
        return data

    def getBestParam(self):
        
            filename = "param_rms.txt"
            #Use file to search best predictions
            param = np.loadtxt(fname = filename,ndmin = 2)   
        
            rms_list = param[:,param.shape[1]-1]
            print("rms_list shape : ",rms_list.shape[0])
            print("self.parameters shape : ", self.parameters.shape[0])
            print("min :: " ,min(rms_list))
            index_best = np.argmin(rms_list)
            print("rms_list[index_best] = " ,param[index_best])
            print("parameters[index_best]  = ",self.parameters[index_best])
            
            return self.parameters[index_best]
            
    
    def updateParamRmsFile(self):
        
        if(self.step%5 == 0):
        
            filename = "param_rms.txt"
            fichier = open(filename,'w')
            for counter,value in enumerate(self.parameters):
                for e in value:
                    fichier.write(str(e)+ " " )
                fichier.write(str( self.getRms(value) ) + "\n")
            fichier.close()
    
        else:
            
            filename = "param_rms.txt"
            if os.path.isfile(filename) :
                fichier = open(filename,'a')
                for param in self.parameters[self.parameters.shape[0]-10:,:]:
                    for e in param :
                        fichier.write(str(e)+ " ") 
                    fichier.write(str( self.getRms(param) ) + "\n")
                fichier.close()
            else:
                print("Error :: no file param_rms.txt in folder. ")
                exit()                    
    
    
    def getBestParamTrueRms(self):
        nomF = "param_rms.txt"
        if os.path.isfile(nomF) :
            rms_min = 10000
            param = np.loadtxt(fname = nomF,ndmin = 2)
            for counter,value in enumerate(param):
                if(value[3] < rms_min):
                    rms_min = value[3]
                    index = counter
            return param[index][1:3]
        else:
            param_0 = np.empty((self.Npar1))
            for i in range(self.Npar1):
                param_0[i] = (self.bornes[i][1]- self.bornes[i][0])*np.random.random_sample() + self.bornes[i][0]
            return param_0

        
        
        
       
            
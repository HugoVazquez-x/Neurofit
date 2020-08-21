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
import time

t0 = time.time()
Nvar=2  # Z,A
Npar2=0 #parameter of correction
Nres=1  #the result

#Read input data
dataset = np.loadtxt(fname = "HalfLifeForecast_inf1000_4.dat")
error = dataset[:,dataset.shape[1]-1]
dataset = dataset[:,:dataset.shape[1]-1]
dataset[:,dataset.shape[1]-2:] = np.log10(dataset[:,dataset.shape[1]-2:])
np.random.shuffle(dataset)

#Creation of a test set 
test_set = np.copy(dataset[:100,:])
dataset = dataset[100:,:]
Exp_test = np.copy(test_set[:,test_set.shape[1]-1:test_set.shape[1]])
Calc_test = np.copy(test_set[:,test_set.shape[1]-2:test_set.shape[1]-1])

test_set[:,test_set.shape[1]-2] = np.copy(test_set[:,test_set.shape[1]-1] - test_set[:,test_set.shape[1]-2])
test_set = np.copy(test_set[:,:test_set.shape[1]-1])
    
Calc = np.copy(dataset[:,dataset.shape[1]-2:dataset.shape[1]-1])
Exp = np.copy(dataset[:,dataset.shape[1]-1:dataset.shape[1]])
print(math.sqrt(sum((Calc- Exp )**2)/Calc.shape[0]))

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
learning_scores = []
test_scores = []

validation_scores_oo = []
validation_scores_eo = []
validation_scores_ee = []
test_scores_oo = []
test_scores_eo = []
test_scores_ee = []

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
    database_eval = np.copy(dataset[nb_data_eval * fold: nb_data_eval * (fold + 1),:])
    database = np.copy( np.concatenate((dataset[:nb_data_eval * fold,:],
                              dataset[nb_data_eval * (fold + 1):,:]),axis=0))
    
    Calc_train = np.copy(database[:,database.shape[1]-2:database.shape[1]-1])
    Exp_train = np.copy(database[:,database.shape[1]-1:database.shape[1]])
    database[:,database.shape[1]-2] = np.copy(database[:,database.shape[1]-1] - database[:,database.shape[1]-2])
    database = np.copy(database[:,:database.shape[1]-1])
    
    Calc_eval = np.copy(database_eval[:,database_eval.shape[1]-2:database_eval.shape[1]-1])
    Exp_eval = np.copy(database_eval[:,database_eval.shape[1]-1:database_eval.shape[1]])
    database_eval[:,database_eval.shape[1]-2] = np.copy(database_eval[:,database_eval.shape[1]-1] - database_eval[:,database_eval.shape[1]-2])
    database_eval = np.copy(database_eval[:,:database_eval.shape[1]-1])
    
    #Decrypte param_code
    if(param_code == "1000"):
        param = database[:,2].reshape(database[:,2].shape[0],1)
        param_eval = database_eval[:,2].reshape(database_eval[:,2].shape[0],1)
        param_test = test_set[:,2].reshape(test_set[:,2].shape[0],1)
        Npar1=1  #bet2
    elif (param_code == "0100"):
        param = database[:,3].reshape(database[:,3].shape[0],1)
        param_eval = database_eval[:,3].reshape(database_eval[:,3].shape[0],1)
        param_test = test_set[:,3].reshape(test_set[:,3].shape[0],1)
        Npar1=1  #Qbet
    elif (param_code == "0010") : 
        param = database[:,4].reshape(database[:,4].shape[0],1)
        param_eval = database_eval[:,4].reshape(database_eval[:,4].shape[0],1)
        param_test = test_set[:,4].reshape(test_set[:,4].shape[0],1)
        Npar1=1  #Jth
    elif ( param_code == "0001") :
        param = database[:,5].reshape(database[:,5].shape[0],1)
        param_eval = database_eval[:,5].reshape(database_eval[:,5].shape[0],1)
        param_test = test_set[:,5].reshape(test_set[:,5].shape[0],1)
        Npar1=1  #Pth
    elif (param_code == "1100" ):
        param = database[:,2:4]
        param_eval = database_eval[:,2:4]
        param_test = test_set[:,2:4]
        Npar1=2  #bet2,Qbet
    elif (param_code == "1010" ):
        param = np.concatenate((database[:,2].reshape(database[:,2].shape[0],1),database[:,4].reshape(database[:,4].shape[0],1)),axis=1)
        param_eval = np.concatenate((database_eval[:,2].reshape(database_eval[:,2].shape[0],1),database_eval[:,4].reshape(database_eval[:,4].shape[0],1)),axis=1)
        param_test = np.concatenate((test_set[:,2].reshape(test_set[:,2].shape[0],1),test_set[:,4].reshape(test_set[:,4].shape[0],1)),axis=1)
        Npar1=2  #bet2,Jth
    elif (param_code == "1001") :
        param = np.concatenate((database[:,2].reshape(database[:,2].shape[0],1),database[:,5].reshape(database[:,4].shape[0],1)),axis=1)
        param_eval = np.concatenate((database_eval[:,2].reshape(database_eval[:,2].shape[0],1),database_eval[:,5].reshape(database_eval[:,4].shape[0],1)),axis=1)
        param_test = np.concatenate((test_set[:,2].reshape(test_set[:,2].shape[0],1),test_set[:,5].reshape(test_set[:,4].shape[0],1)),axis=1)
        Npar1=2  #bet2,Pth
    elif ( param_code == "0110"):
        param = database[:,3:5]
        param_eval = database_eval[:,3:5]
        param_test = test_set[:,3:5]
        Npar1=2  #Qbet,Jth
    elif (param_code == "0101"):
        param = np.concatenate((database[:,3].reshape(database[:,3].shape[0],1),database[:,5].reshape(database[:,4].shape[0],1)),axis=1)
        param_eval = np.concatenate((database_eval[:,3].reshape(database_eval[:,3].shape[0],1),database_eval[:,5].reshape(database_eval[:,4].shape[0],1)),axis=1)
        param_test = np.concatenate((test_set[:,3].reshape(test_set[:,3].shape[0],1),test_set[:,5].reshape(test_set[:,4].shape[0],1)),axis=1)
        Npar1=2  #SQbet,Pth
    elif ( param_code == "0011"):
        param = database[:,4:6]
        param_eval = database_eval[:,4:6]
        param_test = test_set[:,4:6]
        Npar1=2  #Jth,Pth
    elif ( param_code == "1110"):
        param = database[:,2:5]
        param_eval = database_eval[:,2:5]
        param_test = test_set[:,2:5]
        Npar1=3  #bet2,Qbet,Jth
    elif ( param_code == "1101"):
        param = np.concatenate((database[:,2:4],database[:,5].reshape(database[:,5].shape[0],1)),axis=1)
        param_eval = np.concatenate((database_eval[:,2:4],database_eval[:,5].reshape(database_eval[:,5].shape[0],1)),axis=1)
        param_test = np.concatenate((test_set[:,2:4],test_set[:,5].reshape(test_set[:,5].shape[0],1)),axis=1)
        Npar1=3  #bet2,Qbet,Pth
    elif ( param_code == "0111"):
        param = database[:,3:6]
        param_eval = database_eval[:,3:6]
        param_test = test_set[:,3:6]
        Npar1=3  #Qbet,Jth,Pth
    elif (param_code == "1011"):
        param = np.concatenate((database[:,2].reshape(database[:,2].shape[0],1),database[:,4:6]),axis=1)
        param_eval = np.concatenate((database_eval[:,2].reshape(database_eval[:,2].shape[0],1),database_eval[:,4:6]),axis=1)
        param_test = np.concatenate((test_set[:,2].reshape(test_set[:,2].shape[0],1),test_set[:,4:6]),axis=1)
        Npar1=3  #bet2,Jth,Pth
    elif (param_code == "1111"):
        param = database[:,2:6]
        param_eval = database_eval[:,2:6]
        param_test = test_set[:,2:6]
        Npar1=4  #bet2,Qbet,Jth,Pth


    #choose parameters to help the network
    database = np.concatenate((database[:,:2],param,database[:,6:7]), axis=1 )
    database_eval = np.concatenate((database_eval[:,:2],param_eval,database_eval[:,6:7]),axis=1 )
    x_test = np.concatenate((test_set[:,:2],param_test),axis=1)


    nnet = MLNN(database, database_eval, Nvar, Npar1, Npar2, Nres, [0,0], [0,0],additional_param,[0,0],1)
    
    #Parameters of the Neural Networks
    num_hidden_layers = nb_hid_lay
    architecture = arch
    act_func = 'relu'
    successive_fit_numb = 5
    epoch = [2000,2000,2000,2000,2000]
    batch = [3,3,3,3,3]
    
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
    sigma_eval = nnet.model.predict(nnet.x_eval,verbose =0)
    sigma_learning = nnet.model.predict(nnet.x_train,verbose =0)
    x_test_bis = np.copy(x_test)
    x_test_bis -= nnet.mean[:len(nnet.mean)-nnet.Nres]
    x_test_bis /= nnet.std[:len(nnet.std)-nnet.Nres]
    sigma_test = nnet.model.predict(x_test_bis,verbose =0)
    
    #Performances for odd/even
    sigma_test_ee = nnet.model.predict(x_test_bis[ (x_test[:,0]%2==0) & ((x_test[:,1]-x_test[:,0])%2==0) ],verbose =0)
    sigma_test_eo = nnet.model.predict(x_test_bis[ ((x_test[:,0]%2==0) & ((x_test[:,1]-x_test[:,0])%2!=0))   | ((x_test[:,0]%2!=0) & ((x_test[:,1]-x_test[:,0])%2==0)) ],verbose =0)
    sigma_test_oo = nnet.model.predict(x_test_bis[ (x_test[:,0]%2!=0) & ((x_test[:,1]-x_test[:,0])%2!=0) ],verbose =0)
    sigma_eval_ee = nnet.model.predict(nnet.x_eval[ (database_eval[:,0]%2==0) & ((database_eval[:,1]-database_eval[:,0])%2==0) ],verbose =0)
    sigma_eval_eo = nnet.model.predict(nnet.x_eval[ ((database_eval[:,0]%2==0) & ((database_eval[:,1]-database_eval[:,0])%2!=0))  | ((database_eval[:,0]%2!=0) & ((database_eval[:,1]-database_eval[:,0])%2==0)) ],verbose =0)
    sigma_eval_oo = nnet.model.predict(nnet.x_eval[ (database_eval[:,0]%2!=0) & ((database_eval[:,1]-database_eval[:,0])%2!=0) ],verbose =0)    
    
    validation_scores.append(math.sqrt(sum((Calc_eval + sigma_eval - Exp_eval )**2)/sigma_eval.shape[0]))
    learning_scores.append(math.sqrt(sum((Calc_train+ sigma_learning - Exp_train )**2)/sigma_learning.shape[0]))
    test_scores.append(math.sqrt(sum((Calc_test+ sigma_test - Exp_test )**2)/sigma_test.shape[0]))
    
    test_scores_ee.append(math.sqrt(sum((-test_set[ (test_set[:,0]%2==0) & ((x_test[:,1]-test_set[:,0])%2==0)][:,test_set.shape[1]-1:test_set.shape[1]]+ sigma_test_ee )**2)/sigma_test_ee.shape[0]))
    test_scores_eo.append(math.sqrt(sum((-test_set[ ((test_set[:,0]%2==0) & ((x_test[:,1]-test_set[:,0])%2!=0)) | ( (test_set[:,0]%2!=0) & ((x_test[:,1]-test_set[:,0])%2==0))][:,test_set.shape[1]-1:test_set.shape[1] ]+ sigma_test_eo )**2)/sigma_test_eo.shape[0]))
    test_scores_oo.append(math.sqrt(sum((-test_set[ (test_set[:,0]%2!=0) & ((x_test[:,1]-test_set[:,0])%2!=0)][:,test_set.shape[1]-1:test_set.shape[1]]+ sigma_test_oo )**2)/sigma_test_oo.shape[0]))
    validation_scores_ee.append(math.sqrt(sum((-database_eval[ (database_eval[:,0]%2==0) & ((database_eval[:,1]-database_eval[:,0])%2==0)][:,database_eval.shape[1]-1:database_eval.shape[1]]+ sigma_eval_ee )**2)/sigma_eval_ee.shape[0]))
    validation_scores_eo.append(math.sqrt(sum((-database_eval[ ((database_eval[:,0]%2==0) & ((database_eval[:,1]-database_eval[:,0])%2!=0))  | (((database_eval[:,0]%2!=0) & ((database_eval[:,1]-database_eval[:,0])%2==0)))][:,database_eval.shape[1]-1:database_eval.shape[1]]+ sigma_eval_eo )**2)/sigma_eval_eo.shape[0]))
    validation_scores_oo.append(math.sqrt(sum((-database_eval[ (database_eval[:,0]%2!=0) & ((database_eval[:,1]-database_eval[:,0])%2!=0)][:,database_eval.shape[1]-1:database_eval.shape[1]]+ sigma_eval_oo )**2)/sigma_eval_oo.shape[0]))
    
    #Plot performances
    key = list(nnet.trainModel_history[0].history.keys())
    val_mae = key[3]
    
    os.remove("last_model.h5")
    os.remove("weights.best.hdf5")
    os.remove("mean.txt")
    os.remove("std.txt")



validation_score = np.average(validation_scores)
learning_score = np.average(learning_scores)
test_score = np.average(test_scores)
test_score_ee = np.average(test_scores_ee)
test_score_eo = np.average(test_scores_eo)
test_score_oo = np.average(test_scores_oo)
validation_score_ee = np.average(validation_scores_ee)
validation_score_eo = np.average(validation_scores_eo)
validation_score_oo = np.average(validation_scores_oo)

nomFichier = 'GapForecast_results.txt'
if os.path.isfile(nomFichier) :
    fichier = open(nomFichier,'a')
else:
    fichier = open(nomFichier,'w')
fichier.write(param_code +  " "  + str(arch) +  " "  + str(learning_score) 
                         +  " "  + str(validation_score) 
                         +  " "  + str(validation_score_ee)+  "  " + str(validation_score_eo) +  "  " + str(validation_score_oo)
                         +  " "  + str(test_score) 
                         +  " "  + str(test_score_ee) + " " + str(test_score_eo) + " " + str(test_score_oo)
                         +  " "         "\n")
fichier.close()

print(time.time() - t0)









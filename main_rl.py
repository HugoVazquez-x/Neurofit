# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 18:03:05 2020

@author: hugov
"""
import math
import random
import numpy as np
from neurofit_rl import neurofit
import os
import sys

def fct(x,a,b,c):
   return c*math.cos(a*x+b)

#Values to be found
a=2
b=1
c=3

list_pts = np.arange( -5, 5, 0.5 )


# Make of simulated experimental data
exp_values = np.zeros(( list_pts.shape[0], 2 ))

exp_values[:,0] = list_pts

for i in range(exp_values.shape[0]):
	exp_values[i,1] =  fct( exp_values[i,0], a ,b,c)

# add noise
sig=1.
random.seed(30)
exp_values[:,1] += -sig/2. + sig * np.random.rand(exp_values.shape[0])

#I write the data on a file to compare with other fitting method
np.savetxt( "exp_values.txt", exp_values, delimiter = ' ' )

# The user define the boundaries
bornes = np.zeros( ( 3,2  ) )

bornes[0,0] = -5
bornes[0,1] = +5

bornes[1,0] = -5
bornes[1,1] = +5

bornes[2,0] = -5
bornes[2,1] = +5


Nvar=1  # x
Npar1=3  #a,b,c
Npar2=0 #parameter of correction
Nres=1  #the result

#Architecture parameters
nb_hid_lay = int(sys.argv[2])
arch = []
for i in range(3,len(sys.argv)):
    arch.append(int(sys.argv[i]))
    
#Number of data for training
nb_data_train = int(sys.argv[1])

#writting results
nomFichier = 'all_results.txt'
if os.path.isfile(nomFichier) :
    fichier = open(nomFichier,'a')
else:
    fichier = open(nomFichier,'w')
fichier.write("\n")
fichier.write("#" + str(arch))
fichier.write("#" + str(nb_data_train) + "\n")
fichier.close()

nomFichier = 'one_ligne_results.txt'
if os.path.isfile(nomFichier) :
    fichier = open(nomFichier,'a')
else:
    fichier = open(nomFichier,'w')

fichier.write( str(nb_data_train) + " ")
fichier.write( str(len(arch)-1) + " ")
for i in range(len(arch)):
    fichier.write(str(arch[i]) + " ")
fichier.close()

#We start a dataset with 10 random values of x,a,b
database = np.zeros( ( nb_data_train, Nvar + Npar1 + Nres ) )



for i in range( database.shape[0] ):
    random.seed(2*i+85)
    j = random.randrange( 0, list_pts.shape[0] )
    x = list_pts[ j ]
    random.seed(2*i+21)
    a = random.uniform( bornes[0,0], bornes[0,1] )
    random.seed(2*i+25)
    b = random.uniform( bornes[1,0], bornes[1,1] )
    random.seed(4*i+9)
    c = random.uniform( bornes[2,0], bornes[2,1] )
    database[i,0] = x
    database[i,1] = a
    database[i,2] = b
    database[i,3] = c
    database[i,4] = fct(x,a,b,c) - exp_values[j,1]

database_eval = np.zeros( ( math.ceil(nb_data_train*(2/3)), Nvar + Npar1  + Nres ) )

for i in range( database_eval.shape[0] ):
    random.seed(i*10+3)
    j = random.randrange( 0, list_pts.shape[0] )
    x = list_pts[ j ]
    random.seed(i+13)
    a = random.uniform( bornes[0,0], bornes[0,1] )
    random.seed(2*i+4)
    b = random.uniform( bornes[1,0], bornes[1,1] )
    random.seed(2*i+7)
    c = random.uniform( bornes[2,0], bornes[2,1] )
    database_eval[i,0] = x
    database_eval[i,1] = a
    database_eval[i,2] = b
    database_eval[i,3] = c
    database_eval[i,4] = fct(x,a,b,c) - exp_values[j,1]

def additional_param(param_list):
    """
    This function will take param in the same order than in database and compute the additional
    parameter discribe by user.
    The user have to give the number of additional parameter desired and give the functions expected
    in return

    Input : List of parameters (do not change)
    Output : Number of additional parmaters + All the additionnal parameters
    """
    N_add_param = 1

    return N_add_param,param_list[0]*param_list[1]



# start the fitting procedure with neurofit
Nstep= 50
for istep in range( Nstep ):
    print("-----------START OF STEP %2d-----------------------------" % istep)
    
    nomFichier = 'all_results.txt'
    fichier = open(nomFichier,'a')
    fichier.write( str(istep) )
    fichier.close()

    pred = neurofit( database , database_eval , Nvar, Npar1, Npar2, Nres, bornes, list_pts ,additional_param,nb_hid_lay,arch,Nstep,istep,exp_values)
    # pred should be a table of N prediction (line with j,a,b; j being the index of a line of the list_pts table); the last column can give the predicted rms for the value of a and b
    for nb_pred in range(len(pred)):
        #random.seed(nb_pred*6-2)
        j = random.randint(0,list_pts.shape[0]-1)
        x = list_pts[j]
        a =  pred[nb_pred][0]
        b =  pred[nb_pred][1]
        c =  pred[nb_pred][2]
        next_line  = np.zeros( ( 1, Nvar + Npar1 + Npar2 + Nres ) )
        next_line[0,0] = x
        next_line[0,Nres:Nvar + Npar1 ] = pred[nb_pred][0:Npar1]
        #result
        next_line[0,Nvar + Npar1 ] = fct(x,a,b,c) - exp_values[j,1]
        database = np.concatenate(( database , next_line  ),axis=0)
        
fichier.close()

#delete features of current model to start from sratch new model
os.remove("last_model.h5")
os.remove("weights.best.hdf5")
os.remove("mean.txt")
os.remove("std.txt")

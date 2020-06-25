# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 18:03:05 2020

@author: hugov
"""

import random
import numpy as np
from neurofit_rl import neurofit

def fct(x,a,b):
   return a*x + b

#Values to be found
a=2
b=1


list_pts = np.arange( -5, 5, 0.5 )


# Make of simulated experimental data
exp_values = np.zeros(( list_pts.shape[0], 2 ))

exp_values[:,0] = list_pts

for i in range(exp_values.shape[0]):
	exp_values[i,1] =  fct( exp_values[i,0], a ,b)

# add noise
sig=1.
exp_values[:,1] += -sig/2. + sig * np.random.rand(exp_values.shape[0])

#I write the data on a file to compare with other fitting method
np.savetxt( "exp_values.txt", exp_values, delimiter = ' ' )

# The user define the boundaries
bornes = np.zeros( ( 2,2  ) )

bornes[0,0] = -5
bornes[0,1] = +5

bornes[1,0] = -5
bornes[1,1] = +5


Nvar=1  # x
Npar1=2  #a and b
Npar2=1 #parameter of correction
Nres=1  #the result


#We start a dataset with 10 random values of x,a,b
database = np.zeros( ( 500, Nvar + Npar1 + Npar2 + Nres ) )



for i in range( database.shape[0] ):
    j = random.randrange( 0, list_pts.shape[0] )
    x = list_pts[ j ]
    
    a = random.uniform( bornes[0,0], bornes[0,1] )
    b = random.uniform( bornes[1,0], bornes[1,1] )
    database[i,0] = x
    database[i,1] = a
    database[i,2] = b
    database[i,3] = a*x
    database[i,4] = fct(x,a,b) - exp_values[j,1]

database_eval = np.zeros( ( 500, Nvar + Npar1 + Npar2 + Nres ) )

for i in range( database_eval.shape[0] ):
    j = random.randrange( 0, list_pts.shape[0] )
    x = list_pts[ j ]
    a = random.uniform( bornes[0,0], bornes[0,1] )
    b = random.uniform( bornes[1,0], bornes[1,1] )
    database_eval[i,0] = x
    database_eval[i,1] = a
    database_eval[i,2] = b
    database_eval[i,3] = a*x
    database_eval[i,4] = fct(x,a,b) - exp_values[j,1]





# start the fitting procedure with neurofit
Nstep= 1
for istep in range( Nstep ):
    pred = neurofit( database , database_eval , Nvar, Npar1, Npar2, Nres, bornes, list_pts )
    # pred should be a table of N prediction (line with j,a,b; j being the index of a line of the list_pts table); the last column can give the predicted rms for the value of a and b
    #  for i in range( pred.shape[0]):
       #j = int(pred[i,0])
    j = random.randint(0,list_pts.shape[0]-1)
    x = list_pts[j]
    a =  pred[0] 
    b =  pred[1] 
    next_line  = np.zeros( ( 1, Nvar + Npar1 + Npar2 + Nres ) )
    next_line[0,0] = x
    next_line[0,Nres:Nvar + Npar1 ] = pred[0:Npar1]
    #paramètre supplémentaire
    next_line[0,Nvar + Npar1:Nvar + Npar1 + Npar2] = a*x
    #résulat
    next_line[0,Nvar + Npar1 + Npar2] = fct(x,a,b) - exp_values[j,1]
    database = np.concatenate(( database , next_line  ),axis=0)
    print("prédiction des paramètres :" ,pred)
    
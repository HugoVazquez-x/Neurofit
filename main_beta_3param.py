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
import time
import random
import shutil

#Number of

Nvar=2  # N,Z
Npar1=3  #a,b,c
Npar2=0 #parameter of correction
Nres=1  #the result


# The user define the boundaries for each parameter
bornes = np.zeros( ( Npar1,3  ) )

bornes[0,0] = 0.1
bornes[0,1] = +5

bornes[1,0] = 0.1
bornes[1,1] = +5

bornes[2,0] = 0.1
bornes[2,1] = +5

#Architecture parameters
nb_hid_lay = int(sys.argv[1])
arch = []
for i in range(2,len(sys.argv)):
    arch.append(int(sys.argv[i]))

#Read input data
dataset = np.loadtxt(fname = "res2.dat")
Z_A_Yexp_Err = np.loadtxt(fname = "list_pts_exp")
exp_values = np.copy(Z_A_Yexp_Err[:,:3])
list_pts = np.copy(Z_A_Yexp_Err[:,:Nvar])

#Split the dataset  into two set (data_train and data_eval)
nb_data_train = math.ceil(dataset.shape[0]*(3/4))
nb_data_eval = dataset.shape[0] - nb_data_train
database = dataset[:nb_data_train+1,:dataset.shape[1]-1]
database[:,database.shape[1]-2] = np.log10(database[:,database.shape[1]-1]) - np.log10(database[:,database.shape[1]-2])
database = database[:,:database.shape[1]-1]
database_eval = dataset[nb_data_train+1:,:dataset.shape[1]-1]
database_eval[:,database_eval.shape[1]-2] = np.log10(database_eval[:,database_eval.shape[1]-1]) - np.log10(database_eval[:,database_eval.shape[1]-2])
database_eval = database_eval[:,:database_eval.shape[1]-1]

#Writting informations in the output files
nomFichier = 'all_results.txt'
if os.path.isfile(nomFichier) :
    fichier = open(nomFichier,'a')
else:
    fichier = open(nomFichier,'w')
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

def getPhysicalCode(gam0,gam1,fcorr, Z,A,Yexp,Err):
    f_param_lorentz = open("BetaDecay/Package_script/param_lorentz.in",'w') 
    f_param_lorentz.write(str(gam0) + " " + str(gam1) + " " + str(fcorr))
    f_param_lorentz.close()
    shutil.copy("./BetaDecay/Package_script/tminus0.in","./BetaDecay/Package_script/tminus.in")
    f = open("./BetaDecay/Package_script/tminus.in",'a')
    f.write(str(int(Z))+ " " + str(int(A)) + " " + str(Yexp) + " " + str(Err) )
    f.close()
    os.system("run_a2.bat")
    f = open("./BetaDecay/Package_script/t12_d1m_bm.g1.0",'r')
    content = f.readlines()
    f.close()
    line_str = content[1].split(" ")
    line_float = []
    for element in line_str:
            if ( element != ''):
                line_float.append(float(element))
    Ycalc = line_float[5]
    
    return Ycalc

        
def getTrueRms(gam0,gam1,fcorr,Z_A_Yexp_Err):
    print("-------------------COMPUTING TRUE RMS----------------""")
    val = []
    t_0 = time.time()
    f_param_lorentz = open("BetaDecay/Package_script/param_lorentz.in",'w') 
    f_param_lorentz.write(str(gam0) + " " + str(gam1) + " " + str(fcorr))
    f_param_lorentz.close()
    os.system("run_compute_rms2.bat")
    rms_value = np.loadtxt("BetaDecay/Package_script/rms2.dat",ndmin =2)
    for value in rms_value:
        Yexp = value[1]
        Ycalc = value[0]
        val.append(math.log10(Yexp)-math.log10(Ycalc))
    val = np.array(val)
    
    print("-------------------END TRUE RMS COMPUTING : time = ",time.time()-t_0, "s")
    return math.sqrt(sum((val)**2)/val.shape[0])

 

# start the fitting procedure with neurofit
t0_total = time.time()
Nstep= 10
for istep in range( Nstep ):
    t0=time.time()
    print("-----------START OF STEP %2d-----------------------------" % istep)

    nomFichier = 'all_results.txt'
    fichier = open(nomFichier,'a')
    fichier.write( str(istep) )
    fichier.close()

    pred = neurofit( database , database_eval , Nvar, Npar1, Npar2, Nres, bornes, list_pts ,additional_param,nb_hid_lay,arch,Nstep,istep,exp_values)
    # pred should be a table of N prediction (line with j,a,b; j being the index of a line of the list_pts table); the last column can give the predicted rms for the value of a and b
    for ipred in range(len(pred)):

        print(pred[ipred])
        j = random.randint(0,list_pts.shape[0]-1)
        Z_A = list_pts[j]
        gam0 =  pred[ipred][0]
        gam1 = pred[ipred][1]
        fcorr =  pred[ipred][2]
        estimate_rms = pred[ipred][3]
        next_line  = np.zeros( ( 1, Nvar + Npar1 + Npar2 + Nres ) )
        next_line[0,:Nvar] = Z_A
        next_line[0,Nvar:Nvar + Npar1 ] = pred[ipred][0:Npar1]
        

        #Use Physical Code to compute new values
        Yexp = Z_A_Yexp_Err[j][2]
        Err = Z_A_Yexp_Err[j][3]
        Ycalc = getPhysicalCode(gam0,gam1,fcorr,Z_A[0],Z_A[1],Yexp,Err)
        next_line[0,Nvar + Npar1 ] = math.log10(Yexp) - math.log10(Ycalc)
        
        #Add all the predictions in database for next training step
        database = np.concatenate(( database , next_line  ),axis=0)
        

        if(ipred == 0):
            #Use Physical Code to compute the Rms of the best prediction only each 5 steps
#            if ( istep%5 == 0):
            nomF = "best_pred_rms.txt"
            if os.path.isfile(nomF) :
                fichier = open(nomF,'a')
            else:
                fichier = open(nomF,'w')
            fichier.write(str(istep) + " " + str(gam0) + " " + str(gam1) + " " + str(fcorr) + " " + str(getTrueRms(gam0,gam1,fcorr,Z_A_Yexp_Err)) + " " + str(estimate_rms)+ "\n")
            fichier.close()
#            else :
#                nomF = "best_pred_rms.txt"
#                if os.path.isfile(nomF) :
#                    fichier = open(nomF,'a')
#                else:
#                    fichier = open(nomF,'w')
#                fichier.write(str(Nstep) + " " + str(gam0) + " " + str(fcorr)  + " " + str(estimate_rms)+ "\n")
#                fichier.close()
#            
            #Add the first prediction (which is suppose to be the best prediction)
            #in the database_eval. Variable x is changed.                
            x = list_pts[ random.randint( 0, list_pts.shape[0]-1) ]
            while(  (x == next_line[0,:Nvar])[0]&(x == next_line[0,:Nvar])[1] ):
                x = list_pts[ random.randint( 0, list_pts.shape[0]-1) ]
            next_line[0,:Nvar] = x
            database_eval = np.concatenate((database_eval,next_line), axis=0)
    print("epoch time : ",time.time() - t0)

fichier.close()
#fichier = open("best_pred_rms.txt",'a')
#fichier.write(str(Nstep) + " " + str(1) + " " + str(1) + " " + str(getTrueRms(1,1,Z_A_Yexp_Err)) + "\n")
#fichier.close()
print("total computing time : ",time.time()-t0_total)
#delete features of current model to start from sratch new model
os.remove("last_model.h5")
os.remove("weights.best.hdf5")
os.remove("mean.txt")
os.remove("std.txt")
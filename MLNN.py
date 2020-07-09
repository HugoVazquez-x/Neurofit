# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:48:18 2020

@author: hugov
"""

import os
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import math
from minimize import Minimize
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

class MLNN(Minimize):
    def __init__(self,database,database_eval,Nvar,Npar1,Npar2,Nres, bornes, list_pts,additional_param,exp_values):




        #Involve additional parameters in the database
        aux = self.complete_dataset(database,Nvar,Npar1,additional_param)
        data_train = np.concatenate((aux,database[:,Nvar + Npar1:]), axis = 1)
        aux_eval = self.complete_dataset(database_eval,Nvar,Npar1,additional_param)
        data_eval = np.concatenate((aux_eval,database_eval[:,Nvar + Npar1:]), axis = 1)

        #Compute mean and std of the datasets
        if os.path.isfile('mean.txt') and os.path.isfile('std.txt') :
            mean = np.loadtxt(fname = "mean.txt")
            std = np.loadtxt(fname = "std.txt")
        else :
            mean = data_train.mean(axis=0)
            std = data_train.std(axis=0)
            #mean_eval = data_eval.mean(axis=0)
            #std_eval = data_eval.std(axis=0)
            np.savetxt("mean.txt",mean,delimiter=' ')
            np.savetxt("std.txt",std,delimiter=' ')

        #Initialisation of parent class
        Minimize.__init__(self,Nvar, Npar1, Npar2, Nres, bornes, list_pts,mean,std,additional_param)

        """
        database : data use to train the model
        database_eval : data use to eval the model
        Nvar : number of variable in the model
        Npar1 : Number of parameters of the model
        Npar2 : Number of parameters of correction
        Nres : Number of results expected by the model
        list_pts : List of points that the model is expected to predict after training
        """

        self.model = keras.models.Sequential()

        #Split data
        self.x_train = data_train[:,:data_train.shape[1]-Nres]
        self.y_train = data_train[:,data_train.shape[1]-Nres:]
        self.y_train = self.y_train.reshape(self.y_train.shape[0],)

        self.x_eval = data_eval[:,:data_eval.shape[1]-Nres]
        self.y_eval = data_eval[:,data_eval.shape[1]-Nres:]
        self.y_eval = self.y_eval.reshape(self.y_eval.shape[0],)
        
        #Collect list of parameters
        self.parameters = self.x_train[:,Nvar:Nvar+Npar1]
        #Collect exp_values
        self.exp_values = exp_values
        #Normalization of data

        self.x_train -= mean[:len(self.mean)-self.Nres]
        self.x_train /= std[:len(self.std)-self.Nres]
        self.x_eval -= mean[:len(self.mean)-self.Nres]
        self.x_eval /= std[:len(self.std)-self.Nres]

        #Weights
        self.weights = np.ones((self.x_train.shape[0]))

        #ModelHistory in order to draw chart
        self.trainModel_history = []

    def complete_dataset(self,dataset,Nvar,Npar1,additional_param):
        T = []
        for i in range(Nvar + Npar1):
            T.append(dataset[:,i])

        D = additional_param(T)
        param_more  = np.zeros((dataset.shape[0],D[0]))

        for  i in range(D[0]):
            param_more[:,i] = D[i+1]

        dataset = np.concatenate((dataset[:,:Nvar+Npar1],param_more),axis=1)

        return dataset

    def build_model(self,num_hidden_layers,architecture,act_func,output_class=1,optimizers = 'adam',losses = 'mse',metrics = 'mae'):

        """
        Build a a densely connected neural network model from user input
        num_layers : Number of hidden layers (without )
        architecture : List containing the number of unit for each layers ( input layer + hidden layers)
        act_func :Activation function. 'relu', 'sigmoid', 'tanh',...
        in_shape : Dimension of the input vector
        optimizers : SGD, RMSprop, Adam,...
        losses : mse,mael,msle,...
        metrics : mae,acc,...
        output_class : Number of classes in the ouput vector
        """
        #Input Layer
        self.model.add(keras.layers.Dense(architecture[0], activation=act_func,input_shape= (self.x_train.shape[1],)))

        #Hidden Layers
        for i in range(1,num_hidden_layers+1):
            self.model.add(keras.layers.Dense(architecture[i], activation=act_func))

        #Output Layer
        self.model.add(keras.layers.Dense(output_class))
        self.model.compile(optimizer= optimizers, loss = losses, metrics=[metrics])


    def train_model(self,successive_fit_numb,epochs_array,batch_array):
        """
        Train the model in regards to the user input
        At the ends of the training, the best model in regards to monitor function is saved in "best.h5"
        and the model at the end of training is save in "model.h5"
        successive_fit_numb : Number of successive fit during the training
        epochs_array : list of integer representing the number of epochs by fit. Size of epochs must be equal to successive_fit_num
        batch_array : list of integer representing the number of batch by fit. Size of batch must be equal to successive_fit_num
        train_data :
        train_targets :
        test_data :
        test_targets :
        weights : array of numbers that specify how much weight each sample in a batch should have in computing the total loss
        """
        verb = 0
        #A simpler check-point strategy is to save the model weights to the same file, if and only if the validation accuracy improves
        #The best model is saved in file "bestb.h5"
        checkpoint = ModelCheckpoint( monitor='loss', filepath='weights.best.hdf5', save_best_only=True,verbose=verb)
        earlystop = EarlyStopping( monitor="val_mean_absolute_error",min_delta= 0.1,patience=200,verbose=2,mode="min",baseline=None,restore_best_weights=False)
        callbacks_list = [checkpoint,earlystop]
        #? It might be necessary to use a validation dataset during trainig to detect OVERFFITING ?  validation_data = (x_test, y_test)
        print("----------Start of model Training-----------------------")
        for i in range(successive_fit_numb):
            seqM = self.model.fit(self.x_train, self.y_train,
                                  epochs = epochs_array[i] ,
                                  batch_size = batch_array[i],
                                  validation_data = (self.x_eval, self.y_eval),
                                  verbose = verb,
                                  sample_weight = self.weights,
                                  callbacks = callbacks_list)

            self.trainModel_history.append(seqM)
        print("----------End of model Training-------------------------")

        print("----------Loading best weights--------------------------")
        self.model.load_weights("weights.best.hdf5")

        print("----------Saving model as 'last _model.h5'--------------")
        self.model.save('last_model.h5', include_optimizer = True)

        print("----------The model has been saved in 'model.h5'--------")

    def training_view(self,successive_fit_numb):

        """
        Allows to visualize model training history
        This function produce severals charts : a plot of loss and 'mea' on the training dataset over
        training epochs, for each successive fit.
        """

        fig = plt.figure(figsize=(10,10))
        ax =[]
        row = round(math.sqrt(successive_fit_numb))
        col = math.ceil(successive_fit_numb/row)
        col = col*2
        t = 1
        key = list(self.trainModel_history[0].history.keys())
        loss = key[0]
        val_loss = key[2]
        mae = key [1]
        val_mae = key[3]
        for i in range(successive_fit_numb):

            ax.append(fig.add_subplot(row,col,t))
            ax[t-1].plot(self.trainModel_history[i].history[loss])
            ax[t-1].plot(self.trainModel_history[i].history[val_loss])
            plt.title("Fit number %2d of training and validation loss" % i)
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'eval'], loc='upper left')
            t +=1
            ax.append(fig.add_subplot(row,col,t))
            ax[t-1].plot(self.trainModel_history[i].history[mae])
            ax[t-1].plot(self.trainModel_history[i].history[val_mae])
            plt.title("Fit number %2d of training and validation mea" % i)
            plt.ylabel('mae')
            plt.xlabel('epoch')
            plt.legend(['train', 'eval'], loc='upper left')
            t +=1

        plt.show()

    def model_prediction_test(self,test_data):

     predictions = self.model.predict(test_data, verbose=0, batch_size = test_data.shape[0])

     return predictions

    def writing_results(self,name,Pred_results):
        
         key = list(self.trainModel_history[0].history.keys())
         mae = key [1]
         val_mae = key[3]

         nomFichier = name
         if os.path.isfile(nomFichier) :
             fichier = open(nomFichier,'a')
         else:
            fichier = open(nomFichier,'w')
         

         fichier.write(" " + str(self.trainModel_history[0].history[mae][-1]))
         fichier.write(" " + str(self.trainModel_history[0].history[val_mae][-1]) )
         for i in range(len(Pred_results)):
             for j in range(len(Pred_results[i])):
                 fichier.write(" " + str(Pred_results[i][j]))
         fichier.write("\n")
         fichier.close()

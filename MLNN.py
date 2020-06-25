# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:48:18 2020

@author: hugov
"""

import os 
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
from minimize import Minimize
from keras.callbacks import ModelCheckpoint

class MLNN(Minimize):
    def __init__(self,database,database_eval,Nvar,Npar1,Npar2,Nres, bornes, list_pts):
        
        #Force Keras to work with 'float64'
        tf.keras.backend.set_floatx('float64')
        
        #Compute mean in std of the datasets
        mean = database.mean(axis=0)
        mean_eval = database_eval.mean(axis=0)
        std_train = database.std(axis=0)
        std= database_eval.std(axis=0)
        
        #Initialisation of parent class
        Minimize.__init__(self,Nvar, Npar1, Npar2, Nres, bornes, list_pts,mean,std)
        
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
        
        #Normalization of data
        
        
        database -= mean
        database /= std_train
        database_eval -= mean_eval
        database_eval /= std 
        
        
        #Split data
        self.x_train = database[:,:database.shape[1]-Nres]
        self.y_train = database[:,database.shape[1]-Nres:]
        self.y_train = self.y_train.reshape(self.y_train.shape[0],)
        
        self.x_eval = database_eval[:,:database_eval.shape[1]-Nres]
        self.y_eval = database_eval[:,database_eval.shape[1]-Nres:]
        self.y_eval = self.y_eval.reshape(self.y_eval.shape[0],)
        
        #Weights
        self.weights = np.ones((self.x_train.shape[0]))
        
        #ModelHistory in order to draw chart
        self.trainModel_history = []
        

        
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
#        if os.path.isfile('weights.best.hdf5'):
#            self.model.load_weights("weights.best.hdf5")
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
        verb = 2
        #A simpler check-point strategy is to save the model weights to the same file, if and only if the validation accuracy improves
        #The best model is saved in file "bestb.h5"
        checkpoint = ModelCheckpoint( monitor='loss', filepath='weights.best.hdf5', save_best_only=True,verbose=verb)   
        callbacks_list = [checkpoint]
        #? It might be necessary to use a validation dataset during trainig to detect OVERFFITING ?  validation_data = (x_test, y_test)
        print("----------Start of model Training------------")
        for i in range(successive_fit_numb):
            seqM = self.model.fit(self.x_train, self.y_train,
                                  epochs = epochs_array[i] ,
                                  batch_size = batch_array[i],
                                  validation_data = (self.x_eval, self.y_eval), 
                                  verbose = verb, 
                                  sample_weight = self.weights,
                                  callbacks = callbacks_list)
            
            self.trainModel_history.append(seqM)
        print("----------End of model Training---------")    
        #self.model.save('model.h5')    
        #print("----------The model has been saved in 'model.h5'-------------")
        
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
        for i in range(successive_fit_numb):
            
            ax.append(fig.add_subplot(row,col,t))
            ax[t-1].plot(self.trainModel_history[i].history['loss'])
            ax[t-1].plot(self.trainModel_history[i].history['val_loss'])
            plt.title("Fit number %2d of training and validation loss" % i)
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'eval'], loc='upper left')
            t +=1
            ax.append(fig.add_subplot(row,col,t))
            ax[t-1].plot(self.trainModel_history[i].history['mean_absolute_error'])
            ax[t-1].plot(self.trainModel_history[i].history['val_mean_absolute_error'])
            plt.title("Fit number %2d of training and validation mea" % i)
            plt.ylabel('mea')
            plt.xlabel('epoch')
            plt.legend(['train', 'eval'], loc='upper left')
            t +=1
        
        plt.show()
        
    def model_prediction_test(self,test_data):
     
     predictions = self.model.predict(test_data, verbose=0, batch_size = test_data.shape[0])
     
     return predictions
    
    
        
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 09:36:21 2019

@author: Ryan

Modified from https://github.com/RyanCodes44/Machine-Learning-Exercise/blob/master/train.py

-   This modification allows for different penalty weights for your 2 classes, 
    and to have a variable learning schedule

"""

from CNN import CNN
from sklearn.model_selection import KFold
from keras.optimizers import SGD
from keras.utils import plot_model
import pickle

def train_model(train_images, train_labels, n_dim,valid=False, numepochs = 20, folds = 5, shuffle = True,  \
                sgd = SGD(lr=0.1, momentum=0.9, decay=1e-6, nesterov=False), \
                metrics_list = ['accuracy'], callbacks_list = [], sample_weights = None):
    """
    Train CNN
    """    
    
    print("Setting up model...")
    model = CNN((n_dim,n_dim,3))
    plot_model(model, to_file='model.png')
    print(model.summary())
    
    print("Training...")
    
    if valid == True:
        #Set up K-Fold cross-validation due to lack of data
        kf = KFold(folds, shuffle)
        fold = 1
        
        for train_index, valid_index in kf.split(train_images):
            #K-1 used for training, last K fold used for testing/validation
            data_train, data_valid = train_images[train_index], train_images[valid_index]
            labels_train, labels_valid = train_labels[train_index], train_labels[valid_index]
                        
            # Compile model
            
            model.compile(loss='binary_crossentropy', optimizer=sgd, metrics = metrics_list,sample_weight_mode="temporal")
            # Fit the model
            if callbacks_list != []:
                history = model.fit(data_train,labels_train,epochs=numepochs,verbose=1,callbacks=callbacks_list,validation_data=(data_valid,labels_valid),sample_weight = sample_weights)
            else:
                history = model.fit(data_train,labels_train,epochs=numepochs,verbose=1,validation_data=(data_valid,labels_valid),sample_weight = sample_weights)
            model.save("Fold%s.h5" % fold)
            fold += 1
    
    else:
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics = metrics_list,sample_weight_mode="temporal") 
        if callbacks_list != []:
            history = model.fit(train_images,train_labels,epochs=numepochs,verbose=1,callbacks=callbacks_list,sample_weight = sample_weights)
        else:
            history = model.fit(train_images,train_labels,epochs=numepochs,verbose=1,sample_weight = sample_weights)
        model.save("Weights/weights.h5")
    
    with open('trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
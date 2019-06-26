# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 09:46:24 2019

@author: Ryan

Modified from https://github.com/RyanCodes44/Machine-Learning-Exercise/blob/master/run_code.py
"""

from extract_patches import get_data, extract_patches
from train import train_model
from prediction import prediction
import numpy as np
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from CNN import CNN
import math
from PIL import Image

# learning rate schedule 
def step_decay(epoch):
    # initial parameters for learning rate
	initial_lrate = 0.001 #This value will overwrite the initial learning rate you throw into train_model if you have the callback
	drop = 0.25 # use 1 if you don't want to decay the learning rate
	epochs_drop = 5 # how many epochs before you decay the learning rate
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

def run_code(train_image, train_label, n_dim,save_dir,weights,train=False,valid=False,importance = [1,5]):
    
    print("Getting images for training/testing...")
    train_images, train_labels, test_images, test_labels = get_data(train_image,train_label,n_dim,augment=True)
    
    # reshaping / formating of labels 
    train_labels = np.expand_dims(train_labels,axis=2)
    test_labels = np.expand_dims(test_labels,axis=2)
    sample_weights = np.zeros((train_labels.shape[0],train_labels.shape[1]))
    x = np.where(np.squeeze(train_labels) == 1)
    sample_weights[x] = importance[1] # weight of class 1 (i.e. roofs)
    sample_weights[np.where(np.squeeze(train_labels) == 0)] = importance[0] # weight of class 0 (i.e. non-roofs)
    
    if train == True:
        lrate = LearningRateScheduler(step_decay)
        callbacks_list = [lrate]
        train_model(train_images,train_labels,n_dim,valid=valid, numepochs = 50, \
                sgd = SGD(lr=0.001, momentum=0.9, decay=1e-6, nesterov=True), \
                metrics_list = ['accuracy'], callbacks_list = [lrate], sample_weights = sample_weights)
    weights = "Weights/weights.h5"
    
      
    test_pred = prediction(test_images,test_labels,weights,n_dim)
    test_pred = np.reshape(test_pred,(test_images.shape[0],test_images.shape[1],test_images.shape[2],1))

    #Convert confidences to 0 or 1, use 70% threshold
    test_pred = test_pred[:,:,:,0]
    test_pred[np.where(test_pred >= 0.7)] = 1
    test_pred[np.where(test_pred < 0.7)] = 0
    
    for i in range(test_pred.shape[0]):
        pred = test_pred[i]
        pred = np.expand_dims(pred,axis=2)
        image_np = np.multiply(np.array(np.concatenate((pred,pred,pred),axis=2),dtype='uint8'),255)
        image = Image.fromarray(image_np)
        image.save(save_dir + 'test' + str(i) + '.png')
    
    
    
    return test_pred

save_dir = 'Results/'
predictions = run_code('image.tif','labels.tif', 128, save_dir, weights = 'Weights/weights.h5', train=True, valid=False, importance = [1,5])
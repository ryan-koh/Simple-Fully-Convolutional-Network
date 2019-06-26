# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 9:22:15 2019

@author: Ryan
"""


from keras.models import Model
from keras.layers import Input, Reshape
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose

def CNN(input_shape):

    ''' Layers '''
    input_img = Input(input_shape)
    
    Conv1_out = Conv2D(16,(3,3),padding = 'same', activation = 'relu', name = 'conv1')(input_img)
    Conv2_out = Conv2D(16,(5,5),padding = 'same', activation = 'relu', name = 'conv1-2')(Conv1_out)
    
    MPool1 = MaxPooling2D((2,2), name = 'mpool1')(Conv2_out)
    
    Conv3_out = Conv2D(32,(3,3),padding = 'same', activation = 'relu', name = 'conv2')(MPool1)
    Convtrans_out = Conv2DTranspose(32,(2,2),activation = 'relu', strides = (2,2), name = 'convtranspose')(Conv3_out)
    
    Conv4_out = Conv2D(16,(3,3),padding = 'same', activation = 'relu', name = 'conv3')(Convtrans_out)
    out = Conv2D(1,(1,1),padding = 'same', activation = 'sigmoid', name = 'output')(Conv4_out)
    reshaped_out = Reshape((input_shape[0]*input_shape[0],-1))(out)

    ''' Creation of Model '''
    CNN_model = Model(input_img,reshaped_out)
    
    return CNN_model

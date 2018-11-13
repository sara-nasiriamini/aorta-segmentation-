import os
import sys
import random
import warnings

import numpy as np

import matplotlib.pyplot as plt

from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.layers import merge
from keras.layers import add
from keras.layers.merge import concatenate

import tensorflow as tf

def dice_coef(y_true, y_pred):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)


def dice_coef_loss(y_true, y_pred):
	return -dice_coef(y_true, y_pred)
	
def unet2(pretrained_weights = None,input_size = (256,256,1)):
	inputs = Input(input_size)
	
	c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (inputs)
	#print("c1", np.array(c1.shape))
	#c1 = Dropout(0.1) (c1)
	c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
	#print(np.array(c1.shape))
	p1 = MaxPooling2D((2, 2)) (c1)
	#print(np.array(p1.shape))
	
	c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
	#c2 = Dropout(0.1) (c2)
	c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)
	p2 = MaxPooling2D((2, 2)) (c2)

	c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
	#c3 = Dropout(0.2) (c3)
	c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)
	p3 = MaxPooling2D((2, 2)) (c3)

	c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
	#c4 = Dropout(0.2) (c4)
	c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4)
	#print("c4", np.array(c4.shape))
	#d4 = Dropout(0.5)(c4)
	p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
	#print("p4", np.array(p4.shape))

	c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p4)
	#c5 = Dropout(0.3) (c5)
	#print("c5", np.array(c5.shape))
	c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c5)
	#print("c5", np.array(c5.shape))

	#u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same') (c5)
	u6 = Conv2D(512, (2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(c5))
	#print("u6", np.array(u6.shape))
	u6 = concatenate([u6, c4])
	#print("u6", np.array(u6.shape))
	c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)
	#print("c6", np.array(c6.shape))
	#c6 = Dropout(0.2) (c6)
	c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)

	#u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (c6)
	u7 = Conv2D(256, (2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(c6))
	#print("u7", np.array(u7.shape))
	u7 = concatenate([u7, c3])
	c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
	#c7 = Dropout(0.2) (c7)
	c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)

	#u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c7)
	u8 = Conv2D(128, (2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(c7))
	u8 = concatenate([u8, c2])
	c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
	#c8 = Dropout(0.1) (c8)
	c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8)

	#u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c8)
	u9 = Conv2D(64, (2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(c8))
	u9 = concatenate([u9, c1], axis=3)
	c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9)
	#print("c9", np.array(c9.shape))
	#c9 = Dropout(0.1) (c9)
	c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9)
	#print("c9", np.array(c9.shape))
	outputs = Conv2D(3, (1, 1), activation='softmax') (c9)
	#print("outputs", np.array(outputs.shape))
	model = Model(input = inputs, output = outputs)
	#model.compile(optimizer = Adam(lr = 1e-5), loss = 'categorical_crossentropy', metrics = ['accuracy'])
	#model.compile(optimizer = Adam(lr = 5*1e-5), loss = dice_coef_loss, metrics = [dice_coef])
	model.compile(optimizer = Adam(lr = 5*(1e-5)), loss = 'categorical_crossentropy', metrics = [dice_coef])
	
	if(pretrained_weights):
	    model.load_weights(pretrained_weights)
	
	return model
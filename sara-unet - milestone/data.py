from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
from keras.utils import np_utils

def test(test_path, num_image = 30, as_gray = True):
	tests = []
	for i in range(num_image):
	    test = io.imread(os.path.join(test_path,"%d.jpg"%i),as_gray = as_gray)
	    #test = trans.resize(test,(128,128))
	    test = test / 255
	    test = np.reshape(test,test.shape+(1,))
	    tests.append(test)
	return(np.array(tests))


def saveResult(save_path,npyfile):
    for i in range(npyfile.shape[0]):
        img = npyfile[i]
        img_out = np.zeros((img.shape[0],img.shape[1]))
        for j in range(img.shape[0]):
            for k in range(img.shape[1]):
                c = np.argmax(img[j][k])
                img_out[j][k] = color_dict_m[c]
        img_out_1= np.array(img_out, dtype=np.uint8)
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img_out_1)

		
def train(train_path, num_image = 34, as_gray = True):
    images = []
    for i in range(num_image):
	    for j in range(20):
		    k = (24*i) + j
		    if (k < 800):
			    img = io.imread(os.path.join(train_path,"%d.jpg"%k),as_gray = as_gray)
			    #img = trans.resize(img,(128,128))
			    img = img / 255
			    img = np.reshape(img,img.shape+(1,))
			    images.append(img)
    return(np.array(images))		

def label(mask_path, num_image = 34, as_gray = True):
	labels = []
	for i in range(num_image):
		for j in range(20):
			k = (24*i) + j
			if (k < 800):
			    mask = io.imread(os.path.join(mask_path,"%d.png"%k),as_gray = as_gray)
			    #mask = trans.resize(mask,(128,128))
			    new_mask = np.zeros(mask.shape + (3,))
			    mask_val = np.array([0, 127, 254])
			    for z in range(3):
				    new_mask[mask == mask_val[z],z] = 1
			    labels.append(new_mask)
				#new_mask = np_utils.to_categorical(labels,3)
	return(np.array(labels))
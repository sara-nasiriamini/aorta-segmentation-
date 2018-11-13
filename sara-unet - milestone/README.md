# Modified Multi Class U-NET deep learning framework using Keras

U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).

---

## Overview

### Data

Input is 256x256 grayscale images.

### Data augmentation

None


### Model

![/u-net.png](/u-net.png)

This deep neural network is implemented with Keras functional API.

Output from the network is a 256x256x3.

### Training

The model is trained for 5 epochs.

Loss function for the training is catagorial crossentropy.


---

## How to use

### Dependencies

This tutorial depends on the following libraries:

* Tensorflow
* Keras >= 1.0
* Python versions 2.7-3.5

### Run main.py

You will see the predicted results of test image in data/membrane/test

### Or follow notebook trainMyUnet


## About Keras

Keras is a minimalist, highly modular neural networks library, written in Python and capable of running on top of either TensorFlow or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

Use Keras if you need a deep learning library that:

allows for easy and fast prototyping (through total modularity, minimalism, and extensibility).
supports both convolutional networks and recurrent networks, as well as combinations of the two.
supports arbitrary connectivity schemes (including multi-input and multi-output training).
runs seamlessly on CPU and GPU.
Read the documentation [Keras.io](http://keras.io/)

Keras is compatible with: Python 2.7-3.5.

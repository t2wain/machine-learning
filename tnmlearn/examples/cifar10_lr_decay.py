# -*- coding: utf-8 -*-

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tnmlearn.nn.conv import MiniVGGNet
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

def step_decay(epoch):
  # initialize the base initial learning rate, drop factor, and
  # epochs to drop every
  initAlpha = 0.01
  factor = 0.25
  dropEvery = 5
  
  # compute learning rate for the current epoch
  alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))
  
  # return the learning rate
  return float(alpha)
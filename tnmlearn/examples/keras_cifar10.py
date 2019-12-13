# -*- coding: utf-8 -*-

# import the necessary packages

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from tnmlearn.examples import BaseLearningModel
from tnmlearn.datasets import load_cifar10


# %%

class KerasCifar10(BaseLearningModel):
  
  def __init__(self):
    super(KerasCifar10, self).__init__()
    
  
  def getData(self):
    ((self.trainX, self.trainY), 
     (self.testX, self.testY), 
     self.classNames) = load_cifar10(for_dnn=True)
    
    
  def build(self):
    # define the 3072-1024-512-10 architecture using Keras
    model_ = Sequential()
    model_.add(Dense(1024, input_shape=(3072,), activation="relu"))
    model_.add(Dense(512, activation="relu"))
    model_.add(Dense(10, activation="softmax"))
    
    sgd = SGD(0.01)
    model_.compile(loss="categorical_crossentropy", optimizer=sgd,
                  metrics=["accuracy"])
    self.model = model_


  def fit(self):
    return self.fit_(100, 32)


  def evaluate(self):
    self.evaluate_(32)
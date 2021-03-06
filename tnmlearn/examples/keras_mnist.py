# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from tnmlearn.examples import BaseLearningModel
from tnmlearn.datasets import load_mnist


# %%

class KerasMnist(BaseLearningModel):
  
  def __init__(self):
    super(KerasMnist, self).__init__()
    
  
  def getData(self):  
    ((self.trainX, self.trainY), 
     (self.testX, self.testY), 
     self.classNames) = load_mnist()

  
  def build(self):
    # define the 784-256-128-10 architecture using Keras
    self.model = Sequential()
    self.model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
    self.model.add(Dense(128, activation="sigmoid"))
    self.model.add(Dense(10, activation="softmax"))
    
    # train the model using SGD
    print("[INFO] training network...")
    sgd = SGD(0.01)
    self.model.compile(loss="categorical_crossentropy", 
                       optimizer=sgd,
                       metrics=["accuracy"])


  def fit(self):
    return self.fit_(100,128)

    
  def evaluate(self):
    self.evaluate_(128)


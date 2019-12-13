# -*- coding: utf-8 -*-

from tnmlearn.nn.conv import LeNet
from keras.optimizers import SGD
from tnmlearn.examples import BaseLearningModel
from tnmlearn.datasets import load_mnist


# %%

class LeNetMnist(BaseLearningModel):
  
  def __init__(self):
    super(LeNetMnist, self).__init__()
    
  
  def getData(self):
    ((self.trainX, self.trainY), 
     (self.testX, self.testY), 
     self.classNames) = load_mnist(for_cnn=True)


  def build(self):
    # initialize the optimizer and model
    print("[INFO] compiling model...")
    opt = SGD(lr=0.01)
    self.model = LeNet.build(width=28, height=28, depth=1, classes=10)
    self.model.compile(loss="categorical_crossentropy", 
                       optimizer=opt, metrics=["accuracy"])
    
    
  def fit(self):
    return self.fit_(20, 128)

    
  def evaluate(self):
    self.evaluate_(128)
# -*- coding: utf-8 -*-

# import the necessary packages
from tnmlearn.nn.conv import MiniVGGNet
from keras.optimizers import SGD
from tnmlearn.examples import BaseLearningModel
from tnmlearn.datasets import load_cifar10


# %%

class MiniVggNetCifar10(BaseLearningModel):
  
  def __init__(self):
    super(MiniVggNetCifar10, self).__init__()
    
  
  def getData(self):
    ((self.trainX, self.trainY), 
     (self.testX, self.testY), 
     self.classNames) = load_cifar10()
    

  def build(self):
    # initialize the optimizer and model
    print("[INFO] compiling model...")
    opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
    self.model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
    self.model.compile(loss="categorical_crossentropy", optimizer=opt,
                       metrics=["accuracy"])
    

  def fit(self):
    return self.fit_(40, 64)


  def evaluate(self):
    self.evaluate_(64)
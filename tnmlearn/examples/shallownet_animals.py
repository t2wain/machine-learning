# -*- coding: utf-8 -*-

# import the necessary packages
from tnmlearn.preprocessing import ImageToArrayPreprocessor
from tnmlearn.preprocessing import SimplePreprocessor
from tnmlearn.nn.conv import ShallowNet
from keras.optimizers import SGD
from tnmlearn.examples import BaseLearningModel
from tnmlearn.datasets import load_data


# %%

class ShallowNetAnimals(BaseLearningModel):
  
  def __init__(self, datasetpath):
    super(ShallowNetAnimals, self).__init__()
    self.datasetpath = datasetpath
    
  
  def getData(self):
    # initialize the image preprocessors
    sp = SimplePreprocessor(32, 32)
    iap = ImageToArrayPreprocessor()
    preprocessors = [sp, iap]
    ((self.trainX, self.trainY), 
     (self.testX, self.testY), 
     self.classNames) = load_data(self.datasetpath, preprocessors)


  def build(self):
    # initialize the optimizer and model
    print("[INFO] compiling model...")
    opt = SGD(lr=0.005)
    self.model = ShallowNet.build(width=32, height=32, depth=3, classes=2)
    self.model.compile(loss="binary_crossentropy", optimizer=opt,
                       metrics=["accuracy"])
    

  def fit(self):
    return self.fit_(100, 32)


  def evaluate(self):
    self.evaluate_(32)
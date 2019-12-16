# -*- coding: utf-8 -*-

# import the necessary packages
from tnmlearn.preprocessing import ImageToArrayPreprocessor
from tnmlearn.preprocessing import AspectAwarePreprocessor
from tnmlearn.nn.conv import MiniVGGNet
from keras.optimizers import SGD
from tnmlearn.examples import BaseLearningModel
from tnmlearn.datasets import load_data


# %%

class MiniVggNetFlowers17(BaseLearningModel):
  
  def __init__(self, datasetpath):
    super(MiniVggNetFlowers17, self).__init__()
    self.datasetpath = datasetpath


  def getData(self):
    # initialize the image preprocessors
    aap = AspectAwarePreprocessor(64, 64)
    iap = ImageToArrayPreprocessor()
    preprocessors = [aap, iap]
    ((self.trainX, self.trainY), 
     (self.testX, self.testY), 
     self.classNames) = load_data(self.datasetpath, preprocessors)
    
    
  def build(self):
    # initialize the optimizer and model
    print("[INFO] compiling model...")
    opt = SGD(lr=0.05)
    self.model = MiniVGGNet.build(width=64, height=64, depth=3,
                                  classes=len(self.classNames))
    self.model.compile(loss="categorical_crossentropy", optimizer=opt,
                       metrics=["accuracy"])    
    
    
  def fit(self):
    return self.fit_(100, 32)


  def evaluate(self):
    self.evaluate_(32)
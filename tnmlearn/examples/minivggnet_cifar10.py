# -*- coding: utf-8 -*-

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from tnmlearn.nn.conv import MiniVGGNet
from keras.optimizers import SGD
from keras.datasets import cifar10
from tnmlearn.examples import BaseLearningModel


# %%

class MiniVggNetCifar10(BaseLearningModel):
  
  def __init__(self):
    super(MiniVggNetCifar10, self).__init__()
    
  
  def getData(self):
    # load the training and testing data, then scale it into the
    # range [0, 1]
    print("[INFO] loading CIFAR-10 data...")
    ((trainX, trainY), (testX, testY)) = cifar10.load_data()
    
    trainX = trainX.astype("float") / 255.0
    testX = testX.astype("float") / 255.0
    
    # convert the labels from integers to vectors
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)
    
    self.trainX = trainX
    self.trainY = trainY
    self.testX = testX
    self.testY = testY
    
    # initialize the label names for the CIFAR-10 dataset
    self.labelNames = ["airplane", "automobile", "bird", "cat", "deer",
                       "dog", "frog", "horse", "ship", "truck"]
    

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
    self.evaluate_(64,self.labelNames)
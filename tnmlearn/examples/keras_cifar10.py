# -*- coding: utf-8 -*-

# import the necessary packages

from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.datasets import cifar10
from tnmlearn.examples import BaseLearningModel


# %%

class KerasCifar10(BaseLearningModel):
  
  def __init__(self):
    super(KerasCifar10, self).__init__()
    
  
  def getData(self):
    # load the training and testing data, scale it into the range [0, 1],
    # then reshape the design matrix
    print("[INFO] loading CIFAR-10 data...")
    ((trainX_, trainY_), (testX_, testY_)) = cifar10.load_data()
    trainX_ = trainX_.astype("float") / 255.0
    testX_ = testX_.astype("float") / 255.0
    trainX_ = trainX_.reshape((trainX_.shape[0], 3072))
    testX_ = testX_.reshape((testX_.shape[0], 3072))
    
    # convert the labels from integers to vectors
    lb = LabelBinarizer()
    trainY_ = lb.fit_transform(trainY_)
    testY_ = lb.transform(testY_)
    
    self.trainX = trainX_
    self.trainY = trainY_
    self.testX = testX_
    self.testY = testY_
    
    # initialize the label names for the CIFAR-10 dataset
    self.labelNames = ["airplane", "automobile", "bird", "cat", "deer",
      "dog", "frog", "horse", "ship", "truck"]
    
    
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
    self.evaluate_(32, self.labelNames)
# -*- coding: utf-8 -*-

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from sklearn import datasets
from tnmlearn.examples import BaseLearningModel


# %%

class KerasMnist(BaseLearningModel):
  
  def __init__(self):
    super(KerasMnist, self).__init__()
    
  
  def getData(self):
    # grab the MNIST dataset (if this is your first time running this
    # script, the download may take a minute -- the 55MB MNIST dataset
    # will be downloaded)
    print("[INFO] loading MNIST (full) dataset...")
    dataset = datasets.fetch_mldata("MNIST Original")
    
    # scale the raw pixel intensities to the range [0, 1.0], then
    # construct the training and testing splits
    data = dataset.data.astype("float") / 255.0
    (self.trainX, self.testX, self.trainY, self.testY) = train_test_split(data,
      dataset.target, test_size=0.25)
    
    # convert the labels from integers to vectors
    lb = LabelBinarizer()
    self.trainY = lb.fit_transform(self.trainY)
    self.testY = lb.transform(self.testY)
    self.classes_ = lb.classes_ 
    
    
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
    self.evaluate_(128,[str(x) for x in self.classes_])


# -*- coding: utf-8 -*-

from tnmlearn.nn.conv import LeNet
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn import datasets
from keras import backend as K
from tnmlearn.examples import BaseLearningModel


# %%

class LeNetMnist(BaseLearningModel):
  
  def __init__(self):
    super(LeNetMnist, self).__init__()
    
  
  def getData(self):
    # grab the MNIST dataset (if this is your first time using this
    # dataset then the 55MB download may take a minute)
    print("[INFO] accessing MNIST...")
    dataset = datasets.fetch_mldata("MNIST Original")
    data = dataset.data
    
    # if we are using "channels first" ordering, then reshape the
    # design matrix such that the matrix is:
    # num_samples x depth x rows x columns
    if K.image_data_format() == "channels_first":
      data = data.reshape(data.shape[0], 1, 28, 28)
    
    # otherwise, we are using "channels last" ordering, so the design
    # matrix shape should be: num_samples x rows x columns x depth
    else:
      data = data.reshape(data.shape[0], 28, 28, 1)
      
    # scale the input data to the range [0, 1] and perform a train/test
    # split
    (self.trainX, self.testX, self.trainY, self.testY) = train_test_split(data / 255.0,
      dataset.target.astype("int"), test_size=0.25, random_state=42)
    
    # convert the labels from integers to vectors
    le = LabelBinarizer()
    self.trainY = le.fit_transform(self.trainY)
    self.testY = le.transform(self.testY)
    self.classes_ = le.classes_


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
    self.evaluate_(128, [str(x) for x in self.classes_])
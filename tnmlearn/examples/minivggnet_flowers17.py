# -*- coding: utf-8 -*-

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tnmlearn.preprocessing import ImageToArrayPreprocessor
from tnmlearn.preprocessing import AspectAwarePreprocessor
from tnmlearn.datasets import SimpleDatasetLoader
from tnmlearn.nn.conv import MiniVGGNet
from keras.optimizers import SGD
from tnmlearn.other import paths
from tnmlearn.examples import BaseLearningModel
import numpy as np
import os


# %%

class MiniVggNetFlowers17(BaseLearningModel):
  
  def __init__(self, datasetpath):
    super(MiniVggNetFlowers17, self).__init__()
    self.datasetpath = datasetpath


  def getData(self):
    # grab the list of images that we’ll be describing, then extract
    # the class label names from the image paths
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(self.datasetpath))
    classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
    classNames = [str(x) for x in np.unique(classNames)]
    
    # initialize the image preprocessors
    aap = AspectAwarePreprocessor(64, 64)
    iap = ImageToArrayPreprocessor()
    
    # load the dataset from disk then scale the raw pixel intensities
    # to the range [0, 1]
    sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
    (data, labels) = sdl.load(imagePaths, verbose=500)
    data = data.astype("float") / 255.0
    
    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(data, labels,
      test_size=0.25, random_state=42)
    
    # convert the labels from integers to vectors
    trainY = LabelBinarizer().fit_transform(trainY)
    testY = LabelBinarizer().fit_transform(testY)
    
    self.trainX = trainX
    self.testX = testX
    self.trainY = trainY
    self.testY = testY
    self.classNames = classNames
    
    
  def build(self):
    # initialize the optimizer and model
    print("[INFO] compiling model...")
    opt = SGD(lr=0.05)
    model = MiniVGGNet.build(width=64, height=64, depth=3,
                             classes=len(self.classNames))
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])    
    
    
  def fit(self):
    return self.fit_(100, 32)


  def evaluate(self):
    self.evaluate_(32)
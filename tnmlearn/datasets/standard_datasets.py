# -*- coding: utf-8 -*-

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn import datasets
from keras.datasets import cifar10
from keras import backend as K
from tnmlearn.other import paths
from tnmlearn.datasets import SimpleDatasetLoader


# %%

def load_cifar10(for_dnn=False):
    # load the training and testing data, then scale it into the
    # range [0, 1]
    print("[INFO] loading CIFAR-10 data...")
    ((trainX, trainY), (testX, testY)) = cifar10.load_data()
    
    trainX = trainX.astype("float") / 255.0
    testX = testX.astype("float") / 255.0
    
    if for_dnn:
      trainX = trainX.reshape((trainX.shape[0], 3072))
      testX = testX.reshape((testX.shape[0], 3072))
    
    # convert the labels from integers to vectors
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)
       
    # initialize the label names for the CIFAR-10 dataset
    classNames = ["airplane", "automobile", "bird", "cat", "deer",
                  "dog", "frog", "horse", "ship", "truck"]
    
    return ((trainX, trainY), (testX, testY), classNames)
  
  
def load_mnist(for_cnn=False):
  # grab the MNIST dataset (if this is your first time running this
  # script, the download may take a minute -- the 55MB MNIST dataset
  # will be downloaded)
  print("[INFO] loading MNIST (full) dataset...")
  dataset = datasets.fetch_mldata("MNIST Original")
  
  # scale the raw pixel intensities to the range [0, 1.0], then
  # construct the training and testing splits
  data = dataset.data.astype("float") / 255.0
  if for_cnn:
    data = reshape_for_cnn(data, 28, 28, 1)
  (trainX, testX, trainY, testY) = train_test_split(data,
    dataset.target, test_size=0.25)
  
  # convert the labels from integers to vectors
  lb = LabelBinarizer()
  trainY = lb.fit_transform(trainY)
  testY = lb.transform(testY)
  classNames = [str(x) for x in lb.classes_] 
  
  return ((trainX, trainY), (testX, testY), classNames)
  
  
def reshape_for_cnn(data_orig, height, width, channel):
  # if we are using "channels first" ordering, then reshape the
  # design matrix such that the matrix is:
  # num_samples x depth x rows x columns
  if K.image_data_format() == "channels_first":
    data = data_orig.reshape(data_orig.shape[0], channel, height, width)
  
  # otherwise, we are using "channels last" ordering, so the design
  # matrix shape should be: num_samples x rows x columns x depth
  else:
    data = data_orig.reshape(data_orig.shape[0], height, width, channel)

  return data


def load_data(datasetpath, preprocessors):
  # grab the list of images that we’ll be describing
  print("[INFO] loading images...")
  imagePaths = list(paths.list_images(datasetpath))
  
  # load the dataset from disk then scale the raw pixel intensities
  # to the range [0, 1]
  sdl = SimpleDatasetLoader(preprocessors=preprocessors)
  (data, labels) = sdl.load(imagePaths, verbose=500)
  data = data.astype("float") / 255.0
  
  # partition the data into training and testing splits using 75% of
  # the data for training and the remaining 25% for testing
  (trainX, testX, trainY, testY) = train_test_split(
      data, labels, test_size=0.25, random_state=42)
  
  # convert the labels from integers to vectors
  lb = LabelBinarizer()
  lb.fit(labels)
  trainY = lb.transform(trainY)
  classNames = lb.classes_
  if len(classNames) < 3:
    trainY = np.hstack((trainY, 1 - trainY))
    testY = np.hstack((testY, 1 - testY))

  return ((trainX, trainY), (testX, testY), classNames)
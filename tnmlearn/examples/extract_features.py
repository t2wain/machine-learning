# -*- coding: utf-8 -*-

from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from tnmlearn.io import HDF5DatasetWriter
from tnmlearn.other import paths
import numpy as np
import random
import os


def get_data_(imagePaths, labels, batch_size, image_size):
  bs = batch_size
  for i in np.arange(0, len(imagePaths), bs):
    # extract the batch of images and labels, then initialize the
    # list of actual images that will be passed through the network
    # for feature extraction
    batchPaths = imagePaths[i:i + bs]
    batchLabels = labels[i:i + bs]
    batchImages = []
    
    # loop over the images and labels in the current batch
    for (j, imagePath) in enumerate(batchPaths):
      # load the input image using the Keras helper utility
      # while ensuring the image is resized to 224x224 pixels
      image = load_img(imagePath, target_size=image_size)
      image = img_to_array(image)
      
      # preprocess the image by (1) expanding the dimensions and
      # (2) subtracting the mean RGB pixel intensity from the
      # ImageNet dataset
      image = np.expand_dims(image, axis=0)
      image = imagenet_utils.preprocess_input(image)
      
      # add the image to the batch
      batchImages.append(image)
      
      yield (batchLabels, batchImages)


def extract_features(dataset_path, output_file, buffer_size, batch_size):
  bs = batch_size
  imagePaths = list(paths.list_images(dataset_path))
  random.shuffle(imagePaths)
  
  # extract the class labels from the image paths then encode the
  # labels
  labels = [p.split(os.path.sep)[-2] for p in imagePaths]
  le = LabelEncoder()
  labels = le.fit_transform(labels)
  
  # initialize the HDF5 dataset writer, then store the class label
  # names in the dataset
  feature_size = 512 * 7 * 7
  dataset = HDF5DatasetWriter((len(imagePaths), feature_size),
      output_file, dataKey="features", bufSize=buffer_size)
  dataset.storeClassLabels(le.classes_)

  # load the VGG16 network
  print("[INFO] loading network...")
  model = VGG16(weights="imagenet", include_top=False)
  
  image_size = (244,244)
  for i, (batchLabels, batchImages) in enumerate(get_data_(imagePaths, labels, bs, image_size)):
    # pass the images through the network and use the outputs as
    # our actual features
    bs = len(batchLabels)
    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size=bs)

    # reshape the features so that each image is represented by
    # a flattened feature vector of the ‘MaxPooling2D‘ outputs
    features = features.reshape((features.shape[0], feature_size))
    
    # add the features and labels to our HDF5 dataset
    dataset.add(features, batchLabels)
    print("[INFO] writing batch... {}/{}".format(i, bs))

  # close the dataset
  dataset.close()
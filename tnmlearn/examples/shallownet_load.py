# -*- coding: utf-8 -*-

from tnmlearn.preprocessing import ImageToArrayPreprocessor
from tnmlearn.preprocessing import SimplePreprocessor
from tnmlearn.datasets import SimpleDatasetLoader
from keras.models import load_model
from tnmlearn.other import paths
import numpy as np
import cv2


# %%

class ShallowNetLoad:

  def getData(self, datasetpath):
    # initialize the class labels
    self.classLabels = ["cat", "dog", "panda"]
    
    # grab the list of images in the dataset then randomly sample
    # indexes into the image paths list
    print("[INFO] sampling images...")
    self.imagePaths = np.array(list(paths.list_images(datasetpath)))
    idxs = np.random.randint(0, len(self.imagePaths), size=(10,))
    self.imagePaths = self.imagePaths[idxs]
    
    # initialize the image preprocessors
    sp = SimplePreprocessor(32, 32)
    iap = ImageToArrayPreprocessor()
    
    # load the dataset from disk then scale the raw pixel intensities
    # to the range [0, 1]
    sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
    (self.data, self.labels) = sdl.load(self.imagePaths)
    self.data = self.data.astype("float") / 255.0
    
  
  def build(self, modelpath):
    # load the pre-trained network
    print("[INFO] loading pre-trained network...")
    self.model = load_model(modelpath)
    
    
  def predict(self):
    # make predictions on the images
    print("[INFO] predicting...")
    preds = self.model.predict(self.data, batch_size=32).argmax(axis=1)
    
    # loop over the sample images
    for (i, imagePath) in enumerate(self.imagePaths):
      # load the example image, draw the prediction, and display it
      # to our screen
      image = cv2.imread(imagePath)
      cv2.putText(image, "Label: {}".format(self.classLabels[preds[i]]),
                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
      cv2.imshow("Image", image)
      cv2.waitKey(0)
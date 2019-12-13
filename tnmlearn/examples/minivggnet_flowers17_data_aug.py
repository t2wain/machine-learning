# -*- coding: utf-8 -*-

from tnmlearn.examples import MiniVggNetFlowers17
from keras.preprocessing.image import ImageDataGenerator


# %%

class MiniVggNetFlowers17DataAug(MiniVggNetFlowers17):
  
  def __init__(self, datasetpath):
    super(MiniVggNetFlowers17DataAug, self).__init__(datasetpath)
  
  
  def fit(self):
    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
      height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
      horizontal_flip=True, fill_mode="nearest")
    
    # train the network
    print("[INFO] training network...")
    H = self.model.fit_generator(aug.flow(self.trainX, self.trainY, batch_size=32),
      validation_data=(self.testX, self.testY), 
      steps_per_epoch=len(self.trainX) // 32,
      epochs=100, verbose=1)
    
    self.plot_(H, 100)
    return H
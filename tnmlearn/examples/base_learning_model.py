# -*- coding: utf-8 -*-

from sklearn.metrics import classification_report
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import os
from tnmlearn.callbacks import TrainingMonitor


# %%

class BaseLearningModel:
  
  def __init__(self):
    self.callbacks = []


  def buildTrainMonCB_(self, outputpath):
    # construct the set of callbacks
    figPath = os.path.sep.join([outputpath, "{}.png".format(
        os.getpid())])
    jsonPath = os.path.sep.join([outputpath, "{}.json".format(
        os.getpid())])
    self.callbacks.append(TrainingMonitor(figPath, jsonPath=jsonPath))


  def buildModelChkPointCB_(self, weightpath):
    # construct the callback to save only the *best* model to disk
    # based on the validation loss
    fname = os.path.sep.join([weightpath, 
      "weights-{epoch:03d}-{val_loss:.4f}.hdf5"])
    checkpoint = ModelCheckpoint(fname, monitor="val_loss", mode="min",
                                 save_best_only=True, verbose=1)
    self.callbacks.append(checkpoint)


  def fit_(self, epochs=100, batch_size=32):
    # train the model using SGD
    print("[INFO] training network...")
    H = self.model.fit(self.trainX, self.trainY, 
                       callbacks=self.callbacks,
                       validation_data=(self.testX, self.testY),
                       epochs=epochs, batch_size=batch_size)
    self.plot_(H, epochs)
    return H


  def plotModel_(self, outputpath):
    plot_model(self.model, to_file=outputpath, show_shapes=True)
    

  def plot_(self, H, epochs):
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()
    

  def evaluate_(self, batch_size):
    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = self.model.predict(self.testX, batch_size=batch_size)
    print(classification_report(self.testY.argmax(axis=1),
      predictions.argmax(axis=1), target_names=self.classNames))
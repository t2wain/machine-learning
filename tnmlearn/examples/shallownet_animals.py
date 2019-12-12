# -*- coding: utf-8 -*-

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tnmlearn.preprocessing import ImageToArrayPreprocessor
from tnmlearn.preprocessing import SimplePreprocessor
from tnmlearn.datasets import SimpleDatasetLoader
from tnmlearn.nn.conv import ShallowNet
from keras.optimizers import SGD
from tnmlearn.other import paths
from tnmlearn.examples import BaseLearningModel


# %%

class ShallowNetAnimals(BaseLearningModel):
  
  def __init__(self, datasetpath):
    super(ShallowNetAnimals, self).__init__()
    self.datasetpath = datasetpath
    
  
  def getData(self):
    # grab the list of images that we’ll be describing
    print("[INFO] loading images...")
    self.imagePaths = list(paths.list_images(self.datasetpath))
    
    # initialize the image preprocessors
    sp = SimplePreprocessor(32, 32)
    iap = ImageToArrayPreprocessor()
    
    # load the dataset from disk then scale the raw pixel intensities
    # to the range [0, 1]
    sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
    (data, labels) = sdl.load(self.imagePaths, verbose=500)
    data = data.astype("float") / 255.0
    
    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (self.trainX, self.testX, self.trainY, self.testY) = train_test_split(
        data, labels, test_size=0.25, random_state=42)
    
    # convert the labels from integers to vectors
    lb = LabelBinarizer()
    lb.fit(["cat", "dog", "pandas"])
    self.trainY = lb.transform(self.trainY)
    self.testY = lb.transform(self.testY)    


  def build(self):
    # initialize the optimizer and model
    print("[INFO] compiling model...")
    opt = SGD(lr=0.005)
    self.model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
    self.model.compile(loss="categorical_crossentropy", optimizer=opt,
                       metrics=["accuracy"])
    

  def fit(self):
    return self.fit_(100, 32)


  def evaluate(self):
    self.evaluate_(32, ["cat", "dog", "panda"])
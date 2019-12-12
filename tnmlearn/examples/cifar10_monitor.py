# -*- coding: utf-8 -*-

from tnmlearn.examples import MiniVggNetCifar10
from tnmlearn.nn.conv import MiniVGGNet
from keras.optimizers import SGD


# %%

class Cifar10Monitor(MiniVggNetCifar10):
  
  def __init__(self, outputpath):
    super(MiniVggNetCifar10, self).__init__()
    self.outputpath = outputpath

  
  def build(self):
    # initialize the SGD optimizer, but without any learning rate decay
    print("[INFO] compiling model...")
    opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
    self.model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
    self.model.compile(loss="categorical_crossentropy", 
                       optimizer=opt,
                       metrics=["accuracy"])
    self.buildTrainMonCB_(self.outputpath)
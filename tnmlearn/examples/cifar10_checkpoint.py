# -*- coding: utf-8 -*-

from tnmlearn.examples import MiniVggNetCifar10
from tnmlearn.nn.conv import MiniVGGNet
from keras.optimizers import SGD


# %%

class Cifar10Checkpoint(MiniVggNetCifar10):
    
  def __init__(self, outputpath):
    super(Cifar10Checkpoint, self).__init__()
    self.outputpath = outputpath

  
  def build(self):
    # initialize the optimizer and model
    print("[INFO] compiling model...")
    opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
    self.model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
    self.model.compile(loss="categorical_crossentropy", optimizer=opt,
                       metrics=["accuracy"])
    self.buildModelChkPointCB_(self.outputpath)
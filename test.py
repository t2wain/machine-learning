# -*- coding: utf-8 -*-

from tnmlearn.examples import KerasCifar10
from tnmlearn.examples import KerasMnist
from tnmlearn.examples import LeNetMnist
from tnmlearn.examples import ShallowNetAnimals
from tnmlearn.examples import MiniVggNetCifar10
from tnmlearn.examples import Cifar10Monitor
from tnmlearn.examples import Cifar10Checkpoint
from tnmlearn.examples import ImageNetPretrain

# %%

datasetpath = "C:\\Users\\hbad483\\Documents\\Anaconda\\datasets\\animals\\train"
outputpath = "C:\\Users\\hbad483\\Documents\\Anaconda\\output"

# %%

m = Cifar10Checkpoint(outputpath)
m.getData()
m.build()
m.fit()
m.evaluate()

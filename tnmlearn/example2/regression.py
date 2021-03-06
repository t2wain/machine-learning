# -*- coding: utf-8 -*-

import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

X = np.c_[.5, 1].T
y = [.5, 1]
test = np.c_[0, 2].T
#regr = linear_model.LinearRegression()
regr = linear_model.Ridge(alpha=.1)

plt.figure()

np.random.seed(0)
for _ in range(6):
  this_X = .1 * np.random.normal(size=(2,1)) + X
  regr.fit(this_X, y)
  plt.plot(test, regr.predict(test))
  plt.scatter(this_X, y, s=3)
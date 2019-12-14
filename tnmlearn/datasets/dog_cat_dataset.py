# -*- coding: utf-8 -*-

import zipfile
import os
from tnmlearn.other import paths


def unzip_datafile(datafile, destdir):
  os.makedirs(destdir)
  with zipfile.ZipFile(datafile, 'r') as zip_ref:
      zip_ref.extractall(destdir)
  

def split_dog_cat_image_files(traindir):
  catdir = os.path.join(traindir, 'cat')
  dogdir = os.path.join(traindir, 'dog')
  os.mkdir(catdir)
  os.mkdir(dogdir)
  
  imagepaths = [(f, os.path.basename(f)) for f in paths.list_images(traindir)]
  imagepaths = [(f, os.path.join(dogdir if n.startswith('dog') else catdir, n)) 
                for (f, n) in imagepaths]
  
  for (f, fn) in imagepaths:
    os.rename(f, fn)

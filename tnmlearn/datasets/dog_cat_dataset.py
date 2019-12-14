# -*- coding: utf-8 -*-

import zipfile
import os
from tnmlearn.other import paths


def unzip_dataset(trainfile, testfile, datasetdir):
  
  traindir = os.path.join(datasetdir, 'train')
  testdir = os.path.join(datasetdir, 'test')
  os.makedirs(traindir)
  os.makedirs(testdir)
  
  with zipfile.ZipFile(trainfile, 'r') as zip_ref:
      zip_ref.extractall(traindir)
        
  with zipfile.ZipFile(testfile, 'r') as zip_ref:
      zip_ref.extractall(testdir)
      
  return (traindir, testdir)


def split_dog_cat_dataset(traindir):
  imagepaths = list(paths.list_images(traindir))
  filenames = [(f, '/'.join((str.split(f, '/')[:-1])), 
                (str.split(f, '/')[-1])) for f in imagepaths]
  dogfiles = list(filter(lambda f: f[-1].startswith('dog'), filenames))
  catfiles = list(filter(lambda f: f[-1].startswith('cat'), filenames))
  
  dogdir =  os.path.join(dogfiles[0][1],'dog')
  os.mkdir(dogdir)
  for file in dogfiles:
    (f,p,n) = file
    os.rename(f, os.path.join(p,'dog',n))
      
  catdir =  os.path.join(catfiles[0][1],'cat')
  os.mkdir(catdir)
  for file in catfiles:
    (f,p,n) = file
    os.rename(f, os.path.join(p,'cat',n))
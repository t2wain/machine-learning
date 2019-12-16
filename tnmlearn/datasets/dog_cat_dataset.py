# -*- coding: utf-8 -*-

import zipfile
import tarfile
import os
import re
from tnmlearn.other import paths


def unzip_datafile(datafile, destdir):
  os.makedirs(destdir, exist_ok=True)
  with zipfile.ZipFile(datafile, 'r') as zip_ref:
      zip_ref.extractall(destdir)


def extract_file(filePath, to_directory):
    if filePath.endswith('.zip'):
        opener, mode = zipfile.ZipFile, 'r'
    elif filePath.endswith('.tar.gz') or filePath.endswith('.tgz'):
        opener, mode = tarfile.open, 'r:gz'
    elif filePath.endswith('.tar.bz2') or filePath.endswith('.tbz'):
        opener, mode = tarfile.open, 'r:bz2'
    else: 
        return

    os.makedirs(to_directory, exist_ok=True)
    file = opener(filePath, mode)
    try: file.extractall(to_directory)
    finally: file.close()
  

def split_dog_cat_image_files(traindir):
  catdir = os.path.join(traindir, 'cat')
  dogdir = os.path.join(traindir, 'dog')
  os.makedirs(catdir, exist_ok=True)
  os.makedirs(dogdir, exist_ok=True)
  
  imagepaths = [(f, os.path.basename(f)) for f in paths.list_images(traindir)]
  imagepaths = [(f, os.path.join(dogdir if n.startswith('dog') else catdir, n)) 
                for (f, n) in imagepaths]
  
  for (f, fn) in imagepaths:
    os.rename(f, fn)


def split_17flowers(traindir):
  for dir_id in range(17):
    os.makedirs(os.path.join(traindir, 'dir_'+str(dir_id)), exist_ok=True)
    
  imagepaths = [(f, os.path.basename(f)) for f in paths.list_images(traindir)]
  imagepaths = [(f, os.path.join(traindir, 'dir_'+str((int(i)-1)//80), n)) 
                for (f, n) in imagepaths 
                for i in re.findall('(\d{4})', n)]
  
  for (f, fn) in imagepaths:
    os.rename(f, fn)


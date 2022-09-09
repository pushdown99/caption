#
# CNN Transformer in PyTorch and TensorFlow 2 w/ Keras
# tf2/ImageCaption/__main__.py
# Copyright 2022 Haeyeon, Hwang
#
# Main module for the TensorFlow/Keras implementation of Image Captioninh. Run this
# from the root directory, e.g.:
#
# python -m tf2.ImageCaption --help
#

#
# TODO
# ----
# -
#

import os
import platform
import argparse

from .models  import flickr8k
from .models  import vgg16
from .models  import inceptionv3
from .models  import efficientnetb0

os.environ['TF_CPP_MIN_LOG_LEVEL']  = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

print('')
print('________                               _______________')
print('___  __/__________________________________  ____/__  /________      __')
print('__  /  _  _ \_  __ \_  ___/  __ \_  ___/_  /_   __  /_  __ \_ | /| / /')
print('_  /   /  __/  / / /(__  )/ /_/ /  /   _  __/   _  / / /_/ /_ |/ |/ / ')
print('/_/    \___//_/ /_//____/ \____//_/    /_/      /_/  \____/____/|__/  ')
print('')
print('----------------------------------------------------------------------')
print('')

import logging
logging.getLogger('tensorflow').disabled = True

import tensorflow as tf

if __name__ == "__main__":
  parser = argparse.ArgumentParser("ImageCaption")

  # Run-time environment
  cuda_available = tf.test.is_built_with_cuda()
  #gpu_available = tf.test.is_gpu_available(cuda_only = False, min_cuda_compute_capability = None)
  gpu_available = len(tf.config.list_physical_devices('GPU'))

  print("Version        : %s" % (platform.version()))
  print("Python         : %s" % (platform.python_version()))
  print("Tensorflow     : %s" % (tf.__version__))
  print("CUDA Available : %s" % ("yes" if cuda_available else "no"))
  print("GPU Available  : %s" % ("yes" if gpu_available else "no"))
  print("Eager Execution: %s" % ("yes" if tf.executing_eagerly() else "no"))

model   = inceptionv3.cnn()
dataset = flickr8k.dataset()
dataset.load_descriptions('Flickr8k.token.txt', 'descriptions.txt')


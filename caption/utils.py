import tensorflow as tf
import climage
from keras.utils import plot_model
from IPython.display import Image, display

from .models import Flickr8k

def is_notebook() -> bool:
  try:
    shell = get_ipython().__class__.__name__
    print (shell)
    if shell == 'ZMQInteractiveShell':
      return True   # Jupyter notebook or qtconsole
    elif shell == 'TerminalInteractiveShell':
      return False  # Terminal running IPython
    else:
      return False  # Other type (?)
  except NameError:
    return False    # Probably standard Python interpreter

def in_ipynb():
  try:
    cfg = get_ipython().config 
    if cfg['IPKernelApp']['parent_appname'] == 'ipython-notebook':
      return True
    else:
      return False
  except NameError:
    return False

def Display (file):
  if is_notebook():
    display(Image(file))
  else:
    out = climage.convert(file)
    print (out)

def get_model_summary(model: tf.keras.Model) -> str:
  string_list = []
  model.summary(line_length=80, print_fn=lambda x: string_list.append(x))
  return "\n".join(string_list)

def plot(model, filename):
  plot_model(model, to_file=filename, show_shapes=True, show_layer_names=False)
  display(Image(filename))

def get_dataset(data_opt, model_opt, verbose):
  dataset = Flickr8k.Dataset(data_opt, model_opt, verbose)
  data    = dataset.LoadData ()
  return data

def usage():
  import platform
  import os
  import numpy as np

  os.environ['TF_CPP_MIN_LOG_LEVEL']  = '2'
  os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

  import logging
  logging.getLogger('tensorflow').disabled = True

  # Run-time environment
  cuda_available = tf.test.is_built_with_cuda()
  #gpu_available = tf.test.is_gpu_available(cuda_only = False, min_cuda_compute_capability = None)
  gpu_available = len(tf.config.list_physical_devices('GPU'))

  print('')
  print('________                               _______________')
  print('___  __/__________________________________  ____/__  /________      __')
  print('__  /  _  _ \_  __ \_  ___/  __ \_  ___/_  /_   __  /_  __ \_ | /| / /')
  print('_  /   /  __/  / / /(__  )/ /_/ /  /   _  __/   _  / / /_/ /_ |/ |/ / ')
  print('/_/    \___//_/ /_//____/ \____//_/    /_/      /_/  \____/____/|__/  ')
  print('')
  print('----------------------------------------------------------------------')
  print('')

  print("Version        : %s" % (platform.version()))
  print("Python         : %s" % (platform.python_version()))
  print("Tensorflow     : %s" % (tf.__version__))
  print("Numpy          : %s" % (np.__version__))
  print("CUDA Available : %s" % ("yes" if cuda_available else "no"))
  print("GPU Available  : %s" % ("yes" if gpu_available else "no"))
  print("Eager Execution: %s" % ("yes" if tf.executing_eagerly() else "no"))


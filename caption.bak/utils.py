import tensorflow as tf
import climage
from IPython.display import Image, display

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

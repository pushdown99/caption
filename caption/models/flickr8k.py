import os
import string

from numpy import array
from pickle import load
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import add
from tensorflow.keras.callbacks import ModelCheckpoint

class dataset:
  def __init__(self, dir = "datasets/Flickr8k", augment = True, shuffle = True, reduce = True, cache = True, verbose = True):
    if not os.path.exists(dir):
      raise FileNotFoundError("Dataset directory does not exist: %s" % dir)

    self.dir     = dir
    self.verbose = verbose
    self.augment = augment
    self.shuffle = shuffle
    self.cache   = cache
    self.reduce  = reduce

    self.descriptions = None
    self.vocabulary   = None

  # load_doc: load document
  def load_doc (self, filename):
    path = os.path.join(self.dir, filename)
    file = open(path, 'r')
    doc  = file.read()
    file.close()
    return doc
  
  # load_descriptions: load descriptions
  def load_descriptions (self, filename, out):
    doc = self.load_doc (filename)

    descriptions = dict()
    for line in doc.split('\n'):
      tokens = line.split()
      if len(line) < 2:
        continue
      image_id, image_desc = tokens[0], tokens[1:]
      image_id = image_id.split('.')[0]
      image_desc = ' '.join(image_desc)
      if image_id not in descriptions:
        descriptions[image_id] = list()
      descriptions[image_id].append(image_desc)

    if self.verbose:
      print('Loaded: %d ' % len(descriptions))

    #self.descriptions = self.clean_descriptions (descriptions)
    #self.vocabulary   = self.to_vocabulary (descriptions)
    #self.save_descriptions (descriptions, out)

    return descriptions

  # clean_descriptions: clean descriptions
  def clean_descriptions (self, descriptions):
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
      for i in range(len(desc_list)):
        desc = desc_list[i]
        desc = desc.split()
        desc = [word.lower() for word in desc]
        desc = [w.translate(table) for w in desc]
        desc = [word for word in desc if len(word)>1]
        desc = [word for word in desc if word.isalpha()]
        desc_list[i] =  ' '.join(desc)

    return descriptions

  # to_vocabulary: convert the loaded descriptions into a vocabulary of words
  def to_vocabulary (self, descriptions):
    vocabulary = set ()
    for key in descriptions.keys():
      [vocabulary.update(d.split()) for d in descriptions[key]]

    if self.verbose:
      print('Vocab : %d ' % len(vocabulary))

    return vocabulary

  # save_descriptions: save descriptions to file
  def save_descriptions(self, descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
      for desc in desc_list:
        lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

  # load_set: load a pre-defined list of photo identifiers
  def load_set(self, filename):
    doc = load_doc(filename)
    dataset = list()
    for line in doc.split('\n'):
      if len(line) < 1:
        continue
      identifier = line.split('.')[0]
      dataset.append(identifier)

    if self.verbose:
      print('Dataset: %d' % len(dataset))

    return set(dataset)

  # load clean descriptions into memory
  def load_clean_descriptions(self, filename, dataset):
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
      tokens = line.split()
      image_id, image_desc = tokens[0], tokens[1:]
      if image_id in dataset:
        if image_id not in descriptions:
          descriptions[image_id] = list()
        desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
        descriptions[image_id].append(desc)

    if self.verbose:
      print('Train : %d' % len(descriptions))

    return descriptions

  # load photo features
  def load_photo_features(self, filename, dataset):
    all_features = load(open(filename, 'rb'))
    features = {k: all_features[k] for k in dataset}

    if self.verbose:
      print('Train : %d' % len(features))

    return features

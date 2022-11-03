import os
import json
import string

from os import path
from os import makedirs
from os import listdir
from numpy import array
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

from .. import utils

from itertools  import repeat
from os.path    import isfile
from pickle     import dump,load
from tqdm       import tqdm

class Dataset:
  def __init__(self, verbose = True):
    self.verbose = verbose

  def load_doc (self, filename):
    file = open(filename, 'r')
    doc  = file.read()
    file.close()
    return doc
  
  def load_descriptions (self, doc):
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

    return descriptions

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

  def to_vocabulary (self, descriptions):
    vocabulary = set ()
    for key in descriptions.keys():
      [vocabulary.update(d.split()) for d in descriptions[key]]

    return vocabulary

  def save_descriptions(self, descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
      for desc in desc_list:
        lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

  def load_set(self, filename, limits = 0):
    doc = self.load_doc(filename)
    cnt = 0
    dataset = list()
    for line in doc.split('\n'):
      if len(line) < 1:
        continue

      cnt += 1
      identifier = line.split('.')[0]
      dataset.append(identifier)

      if limits != 0 and cnt >= limits:
        break

    return set(dataset)

  def load_coco2014_set(self, directory, limits = 0):
    dataset = list()
    return set(dataset)


  def load_clean_descriptions(self, filename, dataset):
    doc = self.load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
      tokens = line.split()
      image_id, image_desc = tokens[0], tokens[1:]
      if image_id in dataset:
        if image_id not in descriptions:
          descriptions[image_id] = list()
        desc = '<start> ' + ' '.join(image_desc) + ' <end>'
        descriptions[image_id].append(desc)

    return descriptions

  def to_lines(self, descriptions):
    all_desc = list()
    for key in descriptions.keys():
      [all_desc.append(d) for d in descriptions[key]]
    return all_desc

  def create_tokenizer(self, descriptions):
    lines = self.to_lines(descriptions)
    tokenizer = Tokenizer(filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(lines)
    return tokenizer

  def max_len(self, descriptions):
    lines = self.to_lines(descriptions)
    return max(len(d.split()) for d in lines)

  def tokenize_captions(self, tokenizer, name, descriptions):
    caps_lists = list(descriptions.values())
    caps_list = [item for sublist in caps_lists for item in sublist]
    cap_seqs = tokenizer.texts_to_sequences(caps_list)
    cap_seqs = pad_sequences(cap_seqs, padding='post')
    return cap_seqs

  def prepare_mapping_data_flickr8k (self):
    doc = self.load_doc(self.data_opt['images_mapping'])
    descriptions = self.load_descriptions(doc)

    if self.verbose:
      utils.Display (self.data_opt['example_image'])
      print(json.dumps(descriptions[self.data_opt['example_id']], indent=4))

    self.clean_descriptions(descriptions)
    vocabulary = self.to_vocabulary(descriptions)

    filename = 'files/mapping_{}_{}.txt'.format(self.data_opt['dataset_name'],self.model_opt['model_name'])
    self.save_descriptions(descriptions, filename)

  # COCO_train2014_000000247789.jpg
  def prepare_mapping_data_coco2014 (self):
    descriptions = dict()

    filename = "{}/annotations/captions_{}.json".format(self.data_opt['dataset_dir'], "train2014")
    with open(filename, 'r') as f:
      data = json.load(f)

      for val in data['annotations']:
        image_id = "COCO_{}_{:012d}".format("train2014", val['image_id'])
        caption  = val['caption']
        if image_id not in descriptions:
          descriptions[image_id] = list()
        descriptions[image_id].append(caption)

    filename = "{}/annotations/captions_{}.json".format(self.data_opt['dataset_dir'], "val2014")
    with open(filename, 'r') as f:
      data = json.load(f)

      for val in data['annotations']:
        image_id = "COCO_{}_{:012d}".format("val2014", val['image_id'])
        caption  = val['caption']
        if image_id not in descriptions:
          descriptions[image_id] = list()
        descriptions[image_id].append(caption)

    if self.verbose:
      utils.Display (self.data_opt['example_image'])
      print(json.dumps(descriptions[self.data_opt['example_id']], indent=4))

    self.clean_descriptions(descriptions)
    vocabulary = self.to_vocabulary(descriptions)

    filename = 'files/mapping_{}_{}.txt'.format(self.data_opt['dataset_name'],self.model_opt['model_name'])
    self.save_descriptions(descriptions, filename)

  def load_photo_features(self, filename, dataset):
    # load all features
    all_features = load(open(filename, 'rb'))
    # filter features
    features = {k: all_features[k] for k in dataset}
    return features

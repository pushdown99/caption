import os
import json
import string

from os import path
from os import makedirs
from os import listdir
from numpy import array
from pickle import load, dump
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
  def __init__(self, option, verbose = True):
    self.option  = option
    self.verbose = verbose

    for dir in option['dirs']:
      if not path.exists(dir):
        makedirs(dir, exist_ok=True)

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

    if self.verbose:
      print('Loaded: %d ' % len(descriptions))

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

    if self.verbose:
      print('Vocab : %d ' % len(vocabulary))

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

  def load_set(self, filename):
    doc = self.load_doc(filename)
    dataset = list()
    for line in doc.split('\n'):
      if len(line) < 1:
        continue
      identifier = line.split('.')[0]
      dataset.append(identifier)

    #if self.verbose:
    #  print('Dataset: %d' % len(dataset))

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

    #if self.verbose:
    #  print('Descriptions: %d' % len(descriptions))

    return descriptions

  def to_lines(self, descriptions):
    all_desc = list()
    for key in descriptions.keys():
      [all_desc.append(d) for d in descriptions[key]]
    return all_desc

  def create_tokenizer(self, descriptions, top_k):
    lines = self.to_lines(descriptions)
    tokenizer = Tokenizer(num_words=top_k, oov_token="<unk>", filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(lines)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'
    return tokenizer

  def max_len(self, descriptions):
    lines = self.to_lines(descriptions)
    return max(len(d.split()) for d in lines)

  def tokenize_captions(self, tokenizer, descriptions):
    caps_lists = list(descriptions.values())
    caps_list = [item for sublist in caps_lists for item in sublist]
    cap_seqs = tokenizer.texts_to_sequences(caps_list)
    cap_seqs = pad_sequences(cap_seqs, padding='post')
    return cap_seqs

  def prepare_mapping_data (self):
    doc = self.load_doc(self.option['images_mapping'])
    descriptions = self.load_descriptions(doc)

    if self.verbose:
      utils.Display (self.option['example_image'])
      print(json.dumps(descriptions[self.option['example_id']], indent=4))

    self.clean_descriptions(descriptions)
    vocabulary = self.to_vocabulary(descriptions)
    self.save_descriptions(descriptions, self.option['desciption_mapping'])

    if self.verbose:
      print(json.dumps(descriptions[self.option['example_id']], indent=4))

  def GetData (self):
    Data = {
      'dataset_train':      self.dataset_train,
      'dataset_valid':      self.dataset_valid,
      'dataset_test':       self.dataset_test,
      'caption_train':      self.caption_train,
      'caption_valid':      self.caption_valid,
      'caption_test':       self.caption_test,
      'img_name_train':     self.img_name_train,
      'img_name_valid':     self.img_name_valid,
      'img_name_test':      self.img_name_test,
      'train_descriptions': self.train_descriptions,
      'valid_descriptions': self.valid_descriptions,
      'test_descriptions':  self.test_descriptions,
      'vocab_size':         self.vocab_size,
      'max_length':         self.max_length,
      'tokenizer':          self.tokenizer,
    }
    return Data

  def LoadData (self):
    self.prepare_mapping_data ()

    train = self.load_set(self.option['images_train'])
    img_name_train = [x for item in train for x in repeat(item, 5)]
    train_descriptions = self.load_clean_descriptions(self.option['desciption_mapping'], train)

    valid = self.load_set(self.option['images_valid'])
    img_name_valid = [x for item in valid for x in repeat(item, 5)]
    valid_descriptions = self.load_clean_descriptions(self.option['desciption_mapping'], valid)

    test = self.load_set(self.option['images_test'])
    img_name_test = [x for item in test for x in repeat(item, 5)]
    test_descriptions = self.load_clean_descriptions(self.option['desciption_mapping'], test)

    self.dataset_train  = train
    self.dataset_valid  = valid
    self.dataset_test   = test

    self.img_name_train = img_name_train
    self.img_name_valid = img_name_valid
    self.img_name_test  = img_name_test

    self.train_descriptions = train_descriptions
    self.valid_descriptions = valid_descriptions
    self.test_descriptions  = test_descriptions

    filename = self.option['tokenizer'];
    top_k = 5000

    if not isfile(filename):
      tokenizer = self.create_tokenizer(train_descriptions, top_k)
      dump(tokenizer, open(filename, 'wb'))
    else:
      tokenizer = load(open(filename, 'rb'))

    vocab_size = top_k + 1
    max_length = self.max_len(train_descriptions)

    # tokenize captions
    self.caption_train = self.tokenize_captions(tokenizer, train_descriptions)
    self.caption_valid = self.tokenize_captions(tokenizer, valid_descriptions)
    self.caption_test  = self.tokenize_captions(tokenizer, test_descriptions)

    if self.verbose:
      print ('Dataset: {}, {}, {}'.format(len(train_descriptions), len(valid_descriptions), len(test_descriptions)))
      print('Vocabulary Size: %d' % vocab_size)
      print('Description Length: %d' % max_length)
      print (list(tokenizer.word_index.items())[:10])
      print (list(tokenizer.word_index.items())[-10:])
      print(json.dumps(train_descriptions[self.option['example_id']], indent=4))

    self.vocab_size = vocab_size
    self.max_length = max_length
    self.tokenizer  = tokenizer

    return self.GetData()

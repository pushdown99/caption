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

import tensorflow as tf
import collections
import random
 
from tqdm import tqdm
from IPython.display import Image, display

class Dataset:
  def __init__(self, data_opt, model_opt, verbose = True):
    self.data_opt  = data_opt
    self.model_opt = model_opt
    self.verbose   = verbose

    if self.verbose:
      print('* Dataset::__init__()')

    for dir in data_opt['dirs']:
      if not path.exists(dir):
        makedirs(dir, exist_ok=True)

    filename = "{}/annotations/captions_train{}.json".format(self.data_opt['dataset_dir'], 2014)
    self.load_json(filename)

  def load_json (self, filename):
    with open(filename, 'r') as f:
      annotations = json.load(f)

    image_path_to_caption = collections.defaultdict(list)
    for val in annotations['annotations']:
      caption = f"<start> {val['caption']} <end>"
      image_path = "{}/train{}/COCO_train2014_{:012d}.jpg".format(self.data_opt['dataset_dir'], 2014, val['image_id'])
      image_path_to_caption[image_path].append(caption)
    image_paths = list(image_path_to_caption.keys())
    random.shuffle(image_paths)
    train_image_paths = image_paths[:6000]
    print(len(train_image_paths))

    train_captions = []
    img_name_vector = []

    for image_path in train_image_paths:
      caption_list = image_path_to_caption[image_path]
      train_captions.extend(caption_list)
      img_name_vector.extend([image_path] * len(caption_list))

    print(train_captions[0])
    display(Image(img_name_vector[0]))

    self.train_captions = train_captions
    self.img_name_vector = img_name_vector

    # Get unique images
    encode_train = sorted(set(img_name_vector))

    # Feel free to change batch_size according to your system configuration
    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    image_dataset = image_dataset.map(self.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

  def load_image(self, image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

  # Find the maximum length of any caption in our dataset
  def calc_max_length(self, tensor):
    return max(len(t) for t in tensor)

  def GetData (self):
    if self.verbose:
      print('* GetData()')

    Data = {
      'dataset_train':      self.dataset_train,
      'dataset_valid':      self.dataset_valid,
      'dataset_test':       self.dataset_test,
      'img_name_train':     self.img_name_train,
      'img_name_valid':     self.img_name_valid,
      'img_name_test':      self.img_name_test,
      'train_descriptions': self.train_descriptions,
      'valid_descriptions': self.valid_descriptions,
      'test_descriptions':  self.test_descriptions,
      'train_features':     self.train_features,
      'valid_features':     self.valid_features,
      'test_features':      self.test_features,
      'vocab_size':         self.vocab_size,
      'max_length':         self.max_length,
      'tokenizer':          self.tokenizer,
    }
    return Data

  def LoadData (self):
    if self.verbose:
      print('* LoadData()')

"""
  def load_doc (self, filename):
    if self.verbose:
      print('- load_doc({})'.format(filename))

    file = open(filename, 'r')
    doc  = file.read()
    file.close()
    return doc
  
  def load_descriptions (self, doc):
    if self.verbose:
      print('- load_descriptions()')

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
      print('  Loaded: %d ' % len(descriptions))

    return descriptions

  def clean_descriptions (self, descriptions):
    if self.verbose:
      print('- clean_descriptions()')

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
    if self.verbose:
      print('- to_vocabulary()')

    vocabulary = set ()
    for key in descriptions.keys():
      [vocabulary.update(d.split()) for d in descriptions[key]]

    if self.verbose:
      print('  Vocabulary Size: %d ' % len(vocabulary))

    return vocabulary

  def save_descriptions(self, descriptions, filename):
    if self.verbose:
      print('- save_descriptions({})'.format(filename))

    lines = list()
    for key, desc_list in descriptions.items():
      for desc in desc_list:
        lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

  def load_set(self, filename, limits = 0):
    if self.verbose:
      print('- load_set({})'.format(filename))

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

    #if self.verbose:
    #  print('Dataset: %d' % len(dataset))

    return set(dataset)

  def load_clean_descriptions(self, filename, dataset):
    if self.verbose:
      print('- load_clean_descriptions({})'.format(filename))

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
    if self.verbose:
      print('- to_lines()')

    all_desc = list()
    for key in descriptions.keys():
      [all_desc.append(d) for d in descriptions[key]]
    return all_desc

  def create_tokenizer(self, descriptions):
    if self.verbose:
      print('- create_tokenizer()')

    lines = self.to_lines(descriptions)
    tokenizer = Tokenizer(filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(lines)
    return tokenizer

    #lines = self.to_lines(descriptions)
    #tokenizer = Tokenizer(num_words=top_k, oov_token="<unk>", filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    #tokenizer.fit_on_texts(lines)
    #tokenizer.word_index['<pad>'] = 0
    #tokenizer.index_word[0] = '<pad>'
    #return tokenizer

  def max_len(self, descriptions):
    lines = self.to_lines(descriptions)
    return max(len(d.split()) for d in lines)

  def tokenize_captions(self, tokenizer, name, descriptions):
    if self.verbose:
      print('- tokenize_captions({})'.format(name))

    caps_lists = list(descriptions.values())
    caps_list = [item for sublist in caps_lists for item in sublist]
    cap_seqs = tokenizer.texts_to_sequences(caps_list)
    cap_seqs = pad_sequences(cap_seqs, padding='post')
    return cap_seqs

  def prepare_mapping_data (self):
    if self.verbose:
      print('- prepare_mapping_data()')

    doc = self.load_doc(self.data_opt['images_mapping'])
    descriptions = self.load_descriptions(doc)

    if self.verbose:
      utils.Display (self.data_opt['example_image'])
      print(json.dumps(descriptions[self.data_opt['example_id']], indent=4))

    self.clean_descriptions(descriptions)
    vocabulary = self.to_vocabulary(descriptions)

    filename = 'files/mapping_{}_{}.txt'.format(self.data_opt['dataset_name'],self.model_opt['model_name'])
    self.save_descriptions(descriptions, filename)

    if self.verbose:
      print(json.dumps(descriptions[self.data_opt['example_id']], indent=4))

  def load_photo_features(self, filename, dataset):
    # load all features
    all_features = load(open(filename, 'rb'))
    # filter features
    features = {k: all_features[k] for k in dataset}
    return features

  def GetData (self):
    if self.verbose:
      print('* GetData()')

    Data = {
      'dataset_train':      self.dataset_train,
      'dataset_valid':      self.dataset_valid,
      'dataset_test':       self.dataset_test,
      'img_name_train':     self.img_name_train,
      'img_name_valid':     self.img_name_valid,
      'img_name_test':      self.img_name_test,
      'train_descriptions': self.train_descriptions,
      'valid_descriptions': self.valid_descriptions,
      'test_descriptions':  self.test_descriptions,
      'train_features':     self.train_features,
      'valid_features':     self.valid_features,
      'test_features':      self.test_features,
      'vocab_size':         self.vocab_size,
      'max_length':         self.max_length,
      'tokenizer':          self.tokenizer,
    }
    return Data

  def LoadData (self):
    if self.verbose:
      print('* LoadData()')

    self.prepare_mapping_data ()

    filename = 'files/features_{}_{}.pkl'.format(self.data_opt['dataset_name'],self.model_opt['model_name'])
    mapping  = 'files/mapping_{}_{}.txt'.format(self.data_opt['dataset_name'],self.model_opt['model_name'])

    train = self.load_set(self.data_opt['images_train'], self.data_opt['limits_train'])
    img_name_train = [x for item in train for x in repeat(item, 5)]
    train_descriptions = self.load_clean_descriptions(mapping, train)
    train_features = self.load_photo_features(filename, train) 

    if self.verbose:
      print ('  train: dataset({}), description({}), photos({})'.format(len(train), len(train_descriptions), len(train_features)))

    valid = self.load_set(self.data_opt['images_valid'], self.data_opt['limits_valid'])
    img_name_valid = [x for item in valid for x in repeat(item, 5)]
    valid_descriptions = self.load_clean_descriptions(mapping, valid)
    valid_features = self.load_photo_features(filename, valid) 

    if self.verbose:
      print ('  valid: dataset({}), description({}), photos({})'.format(len(valid), len(valid_descriptions), len(valid_features)))

    test = self.load_set(self.data_opt['images_test'], self.data_opt['limits_test'])
    img_name_test = [x for item in test for x in repeat(item, 5)]
    test_descriptions = self.load_clean_descriptions(mapping, test)
    test_features = self.load_photo_features(filename, test) 

    if self.verbose:
      print ('  test : dataset({}), description({}), photos({})'.format(len(test), len(test_descriptions), len(test_features)))

    self.dataset_train  = train
    self.dataset_valid  = valid
    self.dataset_test   = test

    print(train)

    self.img_name_train = img_name_train
    self.img_name_valid = img_name_valid
    self.img_name_test  = img_name_test

    self.train_descriptions = train_descriptions
    self.valid_descriptions = valid_descriptions
    self.test_descriptions  = test_descriptions

    self.train_features = train_features
    self.valid_features = valid_features
    self.test_features  = test_features

    filename = 'files/tokenizer_{}_{}.pkl'.format(self.data_opt['dataset_name'],self.model_opt['model_name'])
    # only create tokenizer if it does not exist
    if not isfile(filename):
        tokenizer = self.create_tokenizer(train_descriptions)
        # save the tokenizer
        dump(tokenizer, open(filename, 'wb'))
    else:
        tokenizer = load(open(filename, 'rb'))
    # define vocabulary size
    vocab_size = len(tokenizer.word_index) + 1
    print('  Vocabulary Size: %d' % vocab_size)
    # determine the maximum sequence length
    max_length = self.max_len(train_descriptions)
    print('  Description Length: %d' % max_length)

    print(list(tokenizer.word_index.items())[:10])
    print(list(tokenizer.word_index.items())[-10:])

    self.vocab_size = vocab_size
    self.max_length = max_length
    self.tokenizer  = tokenizer

    return self.GetData()
"""

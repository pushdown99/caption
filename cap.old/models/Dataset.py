import tensorflow as tf

import os
import gc
import random
import string
import pandas as pd
import collections
from tqdm import tqdm
from pickle import dump,load
from ..utils import Display

class Dataset:
  def __init__(self, config, verbose = True):
    self.verbose = verbose
    self.config  = config

  def preprocessing (self):
    print (len(self.descriptions), len(self.train), len(self.valid), len(self.test))

    key = list(self.descriptions.keys())[0]
    print ('preprocessing: ', self.descriptions[key])

    print (self.train[0])
    Display (self.train[0])

    # dataset
    train_file = 'files/{}_train.pkl'.format(self.config['name'])
    valid_file = 'files/{}_valid.pkl'.format(self.config['name'])
    test_file  = 'files/{}_test.pkl'.format( self.config['name'])

    if not os.path.isfile(train_file):
      dump(self.train, open(train_file, 'wb')) 
      dump(self.valid, open(valid_file, 'wb')) 
      dump(self.test,  open(test_file,  'wb')) 

    # description
    train_file = 'files/{}_train_desc.pkl'.format(self.config['name'])
    valid_file = 'files/{}_valid_desc.pkl'.format(self.config['name'])
    test_file  = 'files/{}_test_desc.pkl'.format (self.config['name'])

    if not os.path.isfile(train_file):
      train_desc = self.load_clean_descriptions(self.descriptions, self.train)
      valid_desc = self.load_clean_descriptions(self.descriptions, self.valid)
      test_desc  = self.load_clean_descriptions(self.descriptions, self.test )
      dump(train_desc, open(train_file, 'wb')) 
      dump(valid_desc, open(valid_file, 'wb')) 
      dump(test_desc,  open(test_file,  'wb'))
    else:
      train_desc = load(open(train_file,'rb'))
      valid_desc = load(open(valid_file,'rb'))
      test_desc  = load(open(test_file, 'rb'))

    key = list(train_desc.keys())[0]
    print (train_desc[key])

    print ('descriptions:train= ', len(train_desc))
    print ('descriptions:valid= ', len(valid_desc))

    # check file
    #Display (list(test_desc.keys())[0])
    #print ('test image: ', list(test_desc.keys())[0])
    #print (list(test_desc.values())[0])

    # prepare tokenizer
    tokenizer_file = 'files/{}_token.pkl'.format(self.config['name'])

    if not os.path.isfile(tokenizer_file):
      tokenizer = self.create_tokenizer(train_desc)
      dump(tokenizer, open(tokenizer_file, 'wb'))
    else:
       tokenizer = load(open(tokenizer_file, 'rb'))

  def load_descriptions (self, df, name, example):
    desc_file = 'files/{}_descriptions.pkl'.format(name)
    self.descriptions = collections.defaultdict(list)
    
    # parse, save & load descriptions
    if not os.path.isfile(desc_file):
      mapping = dict()
      for i, r in tqdm(df.iterrows(), total=df.shape[0]):
        image_id = r['image_id']
        caption  = r['caption']

        if image_id not in mapping:
          mapping[image_id] = list()

        mapping[image_id].append(caption)

      dump(mapping, open(desc_file, 'wb')) 
    else:
      mapping = load(open(desc_file, 'rb'))

    print('Loaded: %d ' % len(mapping))
    # clean descriptions
    self.clean_descriptions(mapping)

    # summarize vocabulary
    vocabulary = self.to_vocabulary(mapping)
    print('Vocabulary Size: %d' % len(vocabulary))

    #Display (example)
    #print ('example image: ',example)
    #print (mapping[example])

    
    #l = list(mapping.items())
    #random.shuffle(l)
    #mapping = dict(l)

    return mapping

  def clean_descriptions(self, descriptions):
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
      for i in range(len(desc_list)):
        desc = desc_list[i]
        # tokenize
        desc = desc.split()
        # convert to lower case
        desc = [word.lower() for word in desc]
        # remove punctuation from each token
        desc = [w.translate(table) for w in desc]
        # remove hanging 's' and 'a'
        desc = [word for word in desc if len(word)>1]
        # remove tokens with numbers in them
        desc = [word for word in desc if word.isalpha()]
        # store as string
        desc_list[i] =  ' '.join(desc)

  # convert the loaded descriptions into a vocabulary of words
  def to_vocabulary(self, descriptions):
    # build a list of all description strings
    all_desc = set()
    for key in descriptions.keys():
      [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc

  # convert a dictionary of clean descriptions to a list of descriptions
  def to_lines(self, descriptions):
    all_desc = list()
    for key in descriptions.keys():
      [all_desc.append(d.replace('<start>','').replace('<end>','')) for d in descriptions[key]]
      #[all_desc.append(d) for d in descriptions[key]]
    return all_desc

  # fit a tokenizer given caption descriptions
  def create_tokenizer (self, descriptions):
    lines = self.to_lines(descriptions)
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(descriptions)
    return tokenizer

  # calculate the length of the description with the most words
  def max_length(self, descriptions):
    lines = self.to_lines(descriptions)
    return max(len(d.split()) for d in lines)

  # create sequences of images, input sequences and output words for an image
  def create_sequences(self, tokenizer, max_length, descriptions, photos, vocab_size):
    X1, X2, y = list(), list(), list()
    # walk through each image identifier
    for key, desc_list in descriptions.items():
      # walk through each description for the image
      for desc in desc_list:
        # encode the sequence
        seq = tokenizer.texts_to_sequences([desc])[0]
        # split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
          # split into input and output pair
          in_seq, out_seq = seq[:i], seq[i]
          # pad input sequence
          in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
          # encode output sequence
          out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
          # store
          X1.append(photos[key][0])
          X2.append(in_seq)
          y.append(out_seq)
    return array(X1), array(X2), array(y)

  # load clean descriptions into memory
  def load_clean_descriptions(self, descriptions, dataset):
    mapping = dict()

    for image_id, desc_list in tqdm(descriptions.items(), total=len(descriptions)):
      if image_id in dataset:
        if image_id not in mapping:
          mapping[image_id] = list()
        for i in range(len(desc_list)):
          caption = '<start> {} <end>'.format(desc_list[i]) 
          mapping[image_id].append(caption)
    return mapping

  def get (self):
    return {
      'name':            self.config['name'],
      'images_dir':      self.config['images_dir'],
      'example_image':   self.config['example_image'],
    }


class coco2014 (Dataset):
  def __init__(self, config, verbose = True):
    super().__init__ (config, verbose)

    train_file   = config['train_file']
    valid_file   = config['valid_file']

    df1 = pd.DataFrame(list(pd.read_json(train_file, lines=True).annotations[0]))
    df1.loc[:,'image_id'] = config['dataset_dir']+'/train2014/COCO_train2014_'+df1['image_id'].map('{:012d}.jpg'.format)

    df2 = pd.DataFrame(list(pd.read_json(valid_file, lines=True).annotations[0]))
    df2.loc[:,'image_id'] = config['dataset_dir']+'/val2014/COCO_val2014_'+df2['image_id'].map('{:012d}.jpg'.format)

    df = pd.concat([df1, df2])
    self.descriptions = self.load_descriptions (df, config['name'], config['example_image'])

    df = pd.DataFrame(list(pd.read_json(train_file, lines=True).annotations[0]))
    df.loc[:,'image_id'] = config['dataset_dir']+'/train2014/COCO_train2014_'+df['image_id'].map('{:012d}.jpg'.format)
    self.train = list(df.loc[:,'image_id']) #self.train = list(set(df.loc[:,'image_id']))
    #random.shuffle(self.train)
    self.train = self.train[:config['train_limit']]

    df = pd.DataFrame(list(pd.read_json(valid_file, lines=True).annotations[0]))
    df.loc[:,'image_id'] = config['dataset_dir']+'/val2014/COCO_val2014_'+df['image_id'].map('{:012d}.jpg'.format)
    df = df.drop_duplicates('image_id')
    df = df.head(config['valid_limit'])

    df_valid = df.sample(frac=config['val_test_split'],random_state=200) #random state is a seed value
    df_test  = df.drop(df_valid.index).sample(frac=1.0)

    self.valid = list(df_valid.image_id)   #self.valid = list(set(df_valid.image_id))
    self.test  = list(df_test.image_id)    #self.test  = list(set(df_test.image_id))

    self.preprocessing()

class flickr8k (Dataset):
  def __init__(self, config, verbose = True):
    super().__init__ (config, verbose)

    caption_file = config['caption_file']
    train_file   = config['train_file']
    valid_file   = config['valid_file']

    df = pd.DataFrame(open(caption_file, 'r').read().strip().split('\n'), columns=['token'])
    df[['image_id', 'idx','caption']] = df['token'].str.split('\t|#', 2, expand=True)
    df.loc[:,'image_id'] = config['images_dir']+'/'+df.image_id
    self.descriptions = self.load_descriptions (df, config['name'], config['example_image'])

    df = pd.DataFrame(open(train_file, 'r').read().strip().split('\n'), columns=['image_id'])
    df.loc[:,'image_id'] = config['images_dir']+'/'+df.image_id
    self.train = list(df.loc[:,'image_id']) #self.train = list(set(df.loc[:,'image_id']))
    #random.shuffle(self.train)
    self.train = self.train[:config['train_limit']]

    df = pd.DataFrame(open(valid_file, 'r').read().strip().split('\n'), columns=['image_id'])
    df.loc[:,'image_id'] = config['images_dir']+'/'+df.image_id

    df_valid = df.sample(frac=config['val_test_split'],random_state=200) #random state is a seed value
    df_test  = df.drop(df_valid.index).sample(frac=1.0)

    self.valid = list(df_valid.image_id)    #self.valid = list(set(df_valid.image_id))
    self.test  = list(df_test.image_id)     #self.test  = list(set(df_test.image_id))

    print ('dataset:train= ', len(self.train))
    print ('dataset:valid= ', len(self.valid))

    self.preprocessing()

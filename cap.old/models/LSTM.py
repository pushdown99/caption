import tensorflow as tf
import numpy as np
import os
import time
import random
import collections
import matplotlib.pyplot as plt

from tqdm import tqdm
from pickle import dump,load
from PIL import Image

from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.efficientnet import EfficientNetB0
from keras.models import Model

from ..utils import plot, Display

class LSTM:
  def __init__ (self, config, verbose=True):
    self.config  = config
    self.verbose = verbose

  def load_image(self, image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (self.config['image_size'], self.config['image_size']))
    if self.config['name'] == 'vgg16':
      img = tf.keras.applications.vgg16.preprocess_input(img)
    else:
      img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

  # extract features from each photo in the directory
  def extract_vgg16_features (self, directory):
    # load the model
    model = VGG16()
    # re-structure the model
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # summarize
    print(model.summary())
    # extract features from each photo
    features = dict()
    for name in tqdm(os.listdir(directory)):
      if not name.endswith(".jpg"):
        continue
      # load an image from file
      filename = directory + '/' + name
      image = tf.keras.utils.load_img(filename, target_size=(224, 224))
      # convert the image pixels to a numpy array
      image = tf.keras.utils.img_to_array(image)
      # reshape data for the model
      image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
      # prepare the image for the VGG model
      image = tf.keras.applications.vgg16.preprocess_input(image)
      # get features
      feature = model.predict(image, verbose=0)
      # get image id
      #image_id = name.split('.')[0]
      # store feature
      #features[image_id] = feature
      features[filename] = feature
      #print('>%s' % filename)
    return features

  def extract_vgg16_features2 (self, dataset):
    # load the model
    model = VGG16()
    # re-structure the model
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # summarize
    print(model.summary())
    #plot(model, 'files/lstm_{}_cnn.png'.format(self.config['name']))

    # extract features from each photo
    features = dict()
    for image_id in tqdm(dataset):
      # load an image from file
      image = tf.keras.utils.load_img(image_id, target_size=(224, 224))
      # convert the image pixels to a numpy array
      image = tf.keras.utils.img_to_array(image)
      # reshape data for the model
      image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
      # prepare the image for the VGG model
      image = tf.keras.applications.vgg16.preprocess_input(image)
      # get features
      feature = model.predict(image, verbose=0)
      # store feature
      features[image_id] = feature
      print('>%s' % image_id)
    return features

  def extract_inceptionv3_features (self, dataset):
    # load the model
    model = VGG16()
    # re-structure the model
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # summarize
    #print(model.summary())
    plot(model, 'files/lstm_{}_cnn.png'.format(self.config['name']))

    # extract features from each photo
    features = dict()
    for image_id in tqdm(dataset):
      # load an image from file
      image = tf.keras.utils.load_img(image_id, target_size=(224, 224))
      # convert the image pixels to a numpy array
      image = tf.keras.utils.img_to_array(image)
      # reshape data for the model
      image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
      # prepare the image for the VGG model
      image = tf.keras.applications.vgg16.preprocess_input(image)
      # get features
      feature = model.predict(image, verbose=0)
      # store feature
      features[image_id] = feature
    return features

  def extract_features (self, dataset):
    if self.config['name'] == 'vgg16':
      #return self.extract_vgg16_features (dataset)
      return self.extract_vgg16_features(self.images_dir)
    else:
      return self.extract_inceptionv3_features (dataset)

  # covert a dictionary of clean descriptions to a list of descriptions
  def to_lines(self, descriptions):
    all_desc = list()
    for key in descriptions.keys():
      [all_desc.append(d) for d in descriptions[key]]
    return all_desc

  # calculate the length of the description with the most words
  def max_length(self, descriptions):
    lines = self.to_lines(descriptions)
    return max(len(d.split()) for d in lines)

  # create sequences of images, input sequences and output words for an image
  def create_sequences(self, tokenizer, max_length, desc_list, photo, vocab_size):
    X1, X2, y = list(), list(), list()
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
        X1.append(photo)
        X2.append(in_seq)
        y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

  # define the captioning model
  def define_model (self, vocab_size, max_length):
    # feature extractor model
    inputs1 = tf.keras.Input(shape=(4096,))
    fe1 = tf.keras.layers.Dropout(0.5)(inputs1)
    fe2 = tf.keras.layers.Dense(256, activation='relu')(fe1)
    # sequence model
    inputs2 = tf.keras.Input(shape=(max_length,))
    se1 = tf.keras.layers.Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = tf.keras.layers.Dropout(0.5)(se1)
    se3 = tf.keras.layers.LSTM(256)(se2)
    # decoder model
    decoder1 = tf.keras.layers.Concatenate()([fe2, se3])
    decoder2 = tf.keras.layers.Dense(256, activation='relu')(decoder1)
    outputs = tf.keras.layers.Dense(vocab_size, activation='softmax')(decoder2)
    # tie it together [image, seq] [word]
    model = tf.keras.models.Model(inputs=[inputs1, inputs2], outputs=outputs)
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # summarize model
    print(model.summary())
    #plot(model, 'files/lstm_{}_model.png'.format(self.config['name']))

    return model

  # data generator, intended to be used in a call to model.fit_generator()
  def data_generator(self, descriptions, photos, tokenizer, max_length, vocab_size):
    # loop for ever over images
    while 1:
      for key, desc_list in descriptions.items():
        print (key, desc_list)
        # retrieve the photo feature
        photo = photos[key][0]
        in_img, in_seq, out_word = self.create_sequences(tokenizer, max_length, desc_list, photo, vocab_size)
        yield ([in_img, in_seq], out_word)

  # load photo features
  def load_photo_features(self, filename, dataset):
    # load all features
    all_features = load(open(filename, 'rb'))
    # filter features
    features = {k: all_features[k] for k in dataset}
    return features

  def load(self, data):
    self.dataset_name  = data['name']
    self.images_dir    = data['images_dir']
    self.example_image = data['example_image']

  def fit (self, epochs=0):
    test_file  = 'files/{}_test.pkl'.format( self.dataset_name)
    train_file = 'files/{}_train.pkl'.format(self.dataset_name)
    valid_file = 'files/{}_valid.pkl'.format(self.dataset_name)

    train = load(open(train_file, 'rb'))
    valid = load(open(valid_file, 'rb'))
    test  = load(open(test_file,  'rb'))

    print ('fit: ',train[0])

    features_file = 'files/lstm_{}_{}_features.pkl'.format(self.config['name'], self.dataset_name)
    if not os.path.isfile(features_file):
      features = self.extract_features (train + valid + test)
      dump(features, open(features_file, 'wb'))
    else:
      features = load(open(features_file, 'rb'))

    train_features = self.load_photo_features(features_file, train)
    valid_features = self.load_photo_features(features_file, valid)

    print ('photos:train= ', len(train_features))
    print ('photos:valid= ', len(valid_features))

    # train the model, run epochs manually and save after each epoch
    train_file = 'files/{}_train_desc.pkl'.format(self.dataset_name)
    valid_file = 'files/{}_valid_desc.pkl'.format(self.dataset_name)

    train_desc = load(open(train_file, 'rb'))
    valid_desc = load(open(valid_file, 'rb'))

    key = list(train_desc.keys())[0]
    print (train_desc[key])


    train_steps = len(train_desc)
    valid_steps = len(valid_desc)

    print('train_steps: ', train_steps, ' valid_steps: ', valid_steps)

    # prepare tokenizer
    filename   = 'files/{}_token.pkl'.format(self.dataset_name)
    tokenizer  = load(open(filename, 'rb'))

    vocab_size = len(tokenizer.word_index) + 1
    max_length = self.max_length(train_desc)
    print('vocabulary size   : %d' % vocab_size)
    print('description length: %d' % max_length)

    # define the model
    model = self.define_model(vocab_size, max_length)

    if epochs == 0:
      epochs = self.config['num_of_epochs']

    for i in range(epochs):
      print('epochs = ', i)
      # create the train data generator
      train_gen = self.data_generator(train_desc, train_features, tokenizer, max_length, vocab_size)
      print (len(train_desc), len(train_features))
      # create the validation data generator
      valid_gen = self.data_generator(valid_desc, valid_features, tokenizer, max_length, vocab_size)
      print (len(valid_desc), len(valid_features))
      # fit for one epoch
      model.fit(train_gen, validation_data=valid_gen, validation_steps=valid_steps, epochs=1, steps_per_epoch=train_steps, verbose=1)
      # save model
      model.save('files/model_' + str(i) + '.h5')

class vgg16(LSTM):
  def __init__ (self, config, verbose=True):
    super().__init__(config, verbose)

class inceptionv3(LSTM):
  def __init__ (self, config, verbose=True):
    super().__init__(config, verbose)

class efficientb0(LSTM):
  def __init__ (self, config, verbose=True):
    super().__init__(config, verbose)

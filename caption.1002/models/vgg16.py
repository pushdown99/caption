import tensorflow as tf
import random
from os import path
from os import makedirs

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

# Prepare Photo Data
from os import listdir
from os.path import isfile
from pickle import dump
from tqdm import tqdm
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt

# Prepare Text Data
import string

# Load Data
from pickle import load

# Encode Text Data
from keras.preprocessing.text import Tokenizer

from keras.utils import plot_model
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, Concatenate
from keras.layers import RepeatVector, TimeDistributed, concatenate, Bidirectional

# Fit Model
import numpy as np
from keras.utils import pad_sequences, to_categorical

# Evaluate Model
from numpy import argmax, argsort
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

# Generate Captions
from IPython.display import Image, display

from .. import utils

# preprocess the image for the model
def preprocess_image (filename, image_size):
  image = load_img(filename, target_size=(image_size, image_size))
  # convert the image pixels to a numpy array
  image = img_to_array(image)
  # reshape data for the model
  image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
  # prepare the image for the model
  image = preprocess_input(image)
  return image

def extract_features (directory, model, image_size):
  # extract features from each photo
  features = dict()
  for name in tqdm(listdir(directory), position=0, leave=True):
    # load an image from file
    filename = directory + '/' + name
    # preprocess the image for the model
    image = preprocess_image(filename, image_size)
    # get features
    feature = model.predict(image, verbose=0)
    # get image id
    image_id = name.split('.')[0]
    # store feature
    features[image_id] = feature
  return features

class v1 (tf.keras.Model):
  def __init__(self, config, data_opt, model_opt, verbose = True):
    super().__init__()

    for dir in data_opt['dirs']:
      if not path.exists(dir):
        makedirs(dir, exist_ok=True)

    #model = VGG16()
    model = VGG16(weights='imagenet', include_top=True, input_shape = (224, 224, 3))
    if verbose:
      print(model.summary())

    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    if verbose:
      print(model.summary())

    filename = 'models/{}.png'.format(model_opt['model_name'])
    utils.plot(model, filename)

    filename = 'files/features_{}_{}.pkl'.format(data_opt['dataset_name'],model_opt['model_name']) 
    # only extract if file does not exist
    if not isfile(filename):
      # extract features from all images
      directory = data_opt['images_dir']
      features  = extract_features(directory, model, model_opt['image_size'])
      # save to file
      dump(features, open(filename, 'wb'))

    example_image = data_opt['example_image']
    display(Image(example_image))
    image = preprocess_image(example_image, model_opt['image_size'])
    plt.imshow(np.squeeze(image))

    self.config    = config
    self.data_opt  = data_opt
    self.model_opt = model_opt
    self.data      = utils.get_dataset(data_opt, model_opt, verbose)

  def Model1(self):

    embedding_dim  = self.config['embedding_dim']
    units          = self.config['units']
    input_size     = self.model_opt['input_size']
    vocab_size     = self.data['vocab_size']
    max_length     = self.data['max_length']
 
    # feature extractor model
    inputs1 = Input(shape=(input_size,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(embedding_dim, activation='relu')(fe1)
    # sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(units)(se2)
    # decoder model
    decoder1 = Concatenate()([fe2, se3])
    decoder2 = Dense(units, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # summarize model
    print(model.summary())

    filename = 'models/rnn_model_1_{}.png'.format(self.model_opt['model_name'])
    utils.plot(model, filename)

    self.model = model
    self.model_name  = 'rnn_model_1'

    return model

  def Model2(self):

    embedding_dim   = self.config['embedding_dim']
    units           = self.config['units']
    input_size      = self.model_opt['input_size']
    vocab_size      = self.data['vocab_size']
    max_length      = self.data['max_length']

    image_input     = Input(shape=(input_size,))
    image_model_1   = Dense(embedding_dim, activation='relu')(image_input)
    image_model     = RepeatVector(max_length)(image_model_1)

    caption_input   = Input(shape=(max_length,))
    # mask_zero: We zero pad inputs to the same length, the zero mask ignores those inputs
    caption_model_1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(caption_input)
    # Since we are going to predict the next word using the previous words, we have to set return_sequences = True.
    caption_model_2 = LSTM(units, return_sequences=True)(caption_model_1)
    caption_model   = TimeDistributed(Dense(embedding_dim))(caption_model_2)

    # Merging the models and creating a softmax classifier
    final_model_1   = concatenate([image_model, caption_model])
    final_model_2   = Bidirectional(LSTM(units, return_sequences=False))(final_model_1)
    final_model_3   = Dense(units, activation='relu')(final_model_2)
    final_model     = Dense(vocab_size, activation='softmax')(final_model_3)

    model = Model(inputs=[image_input, caption_input], outputs=final_model)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # summarize model
    print(model.summary())

    filename = 'models/rnn_model_2_{}.png'.format(self.model_opt['model_name'])
    utils.plot(model, filename)

    self.model = model
    self.model_name  = 'rnn_model_2'

    return model

  # Create sequences of images, input sequences and output words for an image
  def create_sequences(self, tokenizer, max_length, captions_list, image):
    # X1 : input for image features
    # X2 : input for text features
    # y  : output word
    X1, X2, y = list(), list(), list()
    vocab_size = len(tokenizer.word_index) + 1
    # Walk through each caption for the image
    for caption in captions_list:
      # Encode the sequence
      seq = tokenizer.texts_to_sequences([caption])[0]
      # Split one sequence into multiple X,y pairs
      for i in range(1, len(seq)):
        # Split into input and output pair
        in_seq, out_seq = seq[:i], seq[i]
        # Pad input sequence
        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
        # Encode output sequence
        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
        # Store
        X1.append(image)
        X2.append(in_seq)
        y.append(out_seq)
    return X1, X2, y

  # Data generator, intended to be used in a call to model.fit()
  def data_generator(self, images, captions, tokenizer, max_length, batch_size, random_seed):
    # Setting random seed for reproducibility of results
    random.seed(random_seed)
    # Image ids
    image_ids = list(captions.keys())
    _count=0
    while True:
      if _count >= len(image_ids):
        # Generator exceeded or reached the end so restart it
        _count = 0
      # Batch list to store data
      input_img_batch, input_sequence_batch, output_word_batch = list(), list(), list()
      for i in range(_count, min(len(image_ids), _count+batch_size)):
        # Retrieve the image id
        image_id = image_ids[i]
        # Retrieve the image features
        image = images[image_id][0]
        # Retrieve the captions list
        captions_list = captions[image_id]
        # Shuffle captions list
        random.shuffle(captions_list)
        input_img, input_sequence, output_word = self.create_sequences(tokenizer, max_length, captions_list, image)
        # Add to batch
        for j in range(len(input_img)):
          input_img_batch.append(input_img[j])
          input_sequence_batch.append(input_sequence[j])
          output_word_batch.append(output_word[j])
      _count = _count + batch_size
      yield ([np.array(input_img_batch), np.array(input_sequence_batch)], np.array(output_word_batch))

  def Fit(self):
    # define parameters
    num_of_epochs      = self.config['num_of_epochs']
    batch_size         = self.config['batch_size']
    train_descriptions = self.data['train_descriptions']
    valid_descriptions = self.data['valid_descriptions']
    train_features     = self.data['train_features']
    valid_features     = self.data['valid_features']
    tokenizer          = self.data['tokenizer']
    vocab_size         = self.data['vocab_size']
    max_length         = self.data['max_length']
 
    train_length  = len(train_descriptions)
    val_length    = len(valid_descriptions)
    steps_train   = train_length // batch_size

    if train_length % batch_size != 0:
      steps_train = steps_train + 1

    steps_val = val_length // batch_size

    if val_length % batch_size != 0:
      steps_val   = steps_val + 1


    # Setting random seed for reproducibility of results
    random.seed('1000')
    # Shuffle train data
    ids_train = list(train_descriptions.keys())
    random.shuffle(ids_train)
    train_descriptions = {_id: train_descriptions[_id] for _id in ids_train}

    # Create the train data generator
    generator_train = self.data_generator(train_features, train_descriptions, tokenizer, max_length, batch_size, random_seed='1000')
    # Create the validation data generator
    generator_val = self.data_generator(valid_features, valid_descriptions, tokenizer, max_length, batch_size, random_seed='1000')

    # define checkpoint callback
    filename = 'models/{}_{}_ep{}.h5'.format(self.model_name, self.model_opt['model_name'], num_of_epochs)
    checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # define early stopping callback
    early = EarlyStopping(patience=1, verbose=1)

    history = self.model.fit(generator_train,
      epochs           = num_of_epochs,
      steps_per_epoch  = steps_train,
      validation_data  = generator_val,
      validation_steps = steps_val,
      callbacks        = [checkpoint, early],
      verbose          = 1)

    for label in ["loss","val_loss"]:
      plt.plot(history.history[label],label=label)
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.savefig('models/{}_{}_ep{}_loss.png'.format(self.model_name, self.model_opt['model_name'],num_of_epochs))
    plt.show()

  # generate a description for an image
  def generate_desc(self, model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = '<start>'
    # iterate over the whole length of the sequence
    for i in range(max_length):
      # integer encode input sequence
      sequence = tokenizer.texts_to_sequences([in_text])[0]
      # pad input
      sequence = pad_sequences([sequence], maxlen=max_length)
      # predict next word
      yhat = model.predict([photo,sequence], verbose=0)
      # convert probability to integer
      yhat = argmax(yhat)
      # map integer to word
      word = tokenizer.index_word[yhat]
      # stop if we cannot map the word
      if word is None:
        break
      # append as input for generating the next word
      in_text += ' ' + word
      # stop if we predict the end of the sequence
      if word == '<end>':
        break
    return in_text

  # generate a description for an image using beam search
  def generate_desc_beam_search(self, model, tokenizer, photo, max_length, beam_index=3):
    # seed the generation process
    in_text = [['<start>', 0.0]]
    # iterate over the whole length of the sequence
    for i in range(max_length):
      temp = []
      for s in in_text:
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([s[0]])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next words
        preds = model.predict([photo,sequence], verbose=0)
        word_preds = argsort(preds[0])[-beam_index:]
        # get top predictions
        for w in word_preds:
          next_cap, prob = s[0][:], s[1]
          # map integer to word
          word = tokenizer.index_word[w]
          next_cap += ' ' + word
          prob += preds[0][w]
          temp.append([next_cap, prob])

      in_text = temp
      # sorting according to the probabilities
      in_text = sorted(in_text, reverse=False, key=lambda l: l[1])
      # getting the top words
      in_text = in_text[-beam_index:]

    # get last (best) caption text
    in_text = in_text[-1][0]
    caption_list = []
    # remove leftover <end> 
    for w in in_text.split():
        caption_list.append(w)
        if w == '<end>':
            break
    # convert list to string
    caption = ' '.join(caption_list)
    return caption

  def calculate_scores(self, actual, predicted):
    # calculate BLEU score
    smooth = SmoothingFunction().method4
    bleu1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0), smoothing_function=smooth)*100
    bleu2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)*100
    bleu3 = corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0), smoothing_function=smooth)*100
    bleu4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)*100
    print('  BLEU-1: %f' % bleu1)
    print('  BLEU-2: %f' % bleu2)
    print('  BLEU-3: %f' % bleu3)
    print('  BLEU-4: %f' % bleu4)

  # evaluate the skill of the model
  def evaluate_model(self, model, descriptions, features, tokenizer, max_length):
    actual, predicted = list(), list()
    # step over the whole set
    for key, desc_list in tqdm(descriptions.items(), position=0, leave=True):
      # generate description
      yhat = self.generate_desc(model, tokenizer, features[key], max_length)
      # store actual and predicted
      references = [d.split() for d in desc_list]
      actual.append(references)
      predicted.append(yhat.split())
    print('  Sampling:')
    self.calculate_scores(actual, predicted)

  # evaluate the skill of the model
  def evaluate_model_beam_search(self, model, descriptions, features, tokenizer, max_length, beam_index=3):
    actual, predicted = list(), list()
    # step over the whole set
    for key, desc_list in tqdm(descriptions.items(), position=0, leave=True):
      # generate description beam search
      yhat = self.generate_desc_beam_search(model, tokenizer, features[key], max_length, beam_index)
      # store actual and predicted
      references = [d.split() for d in desc_list]
      actual.append(references)
      predicted.append(yhat.split())
    print('  Beam Search k=%d:' % beam_index)
    self.calculate_scores(actual, predicted)

  def Evaluate (self):
    model             = self.model
    num_of_epochs     = self.config['num_of_epochs']
    batch_size        = self.config['batch_size']
    test_descriptions = self.data['test_descriptions']
    test_features     = self.data['test_features']
    tokenizer         = self.data['tokenizer']
    max_length        = self.data['max_length']

    filename = 'models/{}_{}_ep{}.h5'.format(self.model_name, self.model_opt['model_name'], num_of_epochs)
    model = load_model(filename)
    self.evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)

  def EvaluateBeamSearch (self, beam_index=3):
    model             = self.model
    num_of_epochs     = self.config['num_of_epochs']
    batch_size        = self.config['batch_size']
    test_descriptions = self.data['test_descriptions']
    test_features     = self.data['test_features']
    tokenizer         = self.data['tokenizer']
    max_length        = self.data['max_length']

    filename = 'models/{}_{}_ep{}.h5'.format(self.model_name, self.model_opt['model_name'], num_of_epochs)
    model = load_model(filename)
    self.evaluate_model_beam_search(model, test_descriptions, test_features, tokenizer, max_length, beam_index)

  def clean_caption(self, caption):
    # split caption words
    caption_list = caption.split()
    # remove <start> and <end>
    caption_list = caption_list[1:len(caption_list)-1]
    # convert list to string
    caption = ' '.join(caption_list)
    return caption

  def generate_captions(self, model, descriptions, features, tokenizer, max_length, image_size, count):
    c = 0
    for key, desc_list in descriptions.items():
      # load an image from file
      filename = self.data_opt['images_dir']+'/' + key + '.jpg'
      #diplay image
      display(Image(filename))
      # print original descriptions
      for i, desc in enumerate(desc_list):
        print('  Original ' + str(i+1) + ': ' + self.clean_caption(desc_list[i]))
      # generate descriptions
      desc = self.generate_desc(model, tokenizer, features[key], max_length)
      desc_beam_3 = self.generate_desc_beam_search(model, tokenizer, features[key], max_length, beam_index=3)
      desc_beam_5 = self.generate_desc_beam_search(model, tokenizer, features[key], max_length, beam_index=5)
      # calculate BLEU-1 scores
      references = [d.split() for d in desc_list]
      smooth = SmoothingFunction().method4
      desc_bleu = sentence_bleu(references, desc.split(), weights=(1.0, 0, 0, 0), smoothing_function=smooth)*100
      desc_beam_3_bleu = sentence_bleu(references, desc_beam_3.split(), weights=(1.0, 0, 0, 0), smoothing_function=smooth)*100
      desc_beam_5_bleu = sentence_bleu(references, desc_beam_5.split(), weights=(1.0, 0, 0, 0), smoothing_function=smooth)*100
      # print descriptions with scores
      print('  Sampling (BLEU-1: %f): %s' % (desc_bleu, self.clean_caption(desc)))
      print('  Beam Search k=3 (BLEU-1: %f): %s' % (desc_beam_3_bleu, self.clean_caption(desc_beam_3)))
      print('  Beam Search k=5 (BLEU-1: %f): %s' % (desc_beam_5_bleu, self.clean_caption(desc_beam_5)))
      c += 1
      if c == count:
        break

  def Generate(self):
    num_of_epochs     = self.config['num_of_epochs']
    test_descriptions = self.data['test_descriptions']
    test_features     = self.data['test_features']
    max_length        = self.data['max_length']

    filename = 'files/tokenizer_{}_{}.pkl'.format(self.data_opt['dataset_name'],self.model_opt['model_name'])
    tokenizer = load(open(filename, 'rb'))

    filename = 'models/{}_{}_ep{}.h5'.format(self.model_name, self.model_opt['model_name'], num_of_epochs)
    model = load_model(filename)

    self.generate_captions(model, test_descriptions, test_features, tokenizer, max_length, self.model_opt['image_size'], 10)

class v2 (tf.keras.Model):
  def __init__(self, config, data_opt, model_opt, verbose = True):
    super().__init__()

    for dir in data_opt['dirs']:
      if not path.exists(dir):
        makedirs(dir, exist_ok=True)

    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    if verbose:
      print(model.summary())

    filename = 'models/rnn_model_v2_{}.png'.format(model_opt['model_name'])
    utils.plot(model, filename)

    self.data      = utils.get_dataset(data_opt, model_opt, verbose)

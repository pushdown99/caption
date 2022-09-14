import tensorflow as tf
import numpy as np
import time

from os import path
from os import makedirs
from tqdm import tqdm
from pickle import load, dump
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from PIL import Image
from IPython.display import display

from ..       import utils

class BahdanauAttention (tf.keras.Model):
  def __init__(self, units):
    super().__init__()

    self.units  = units
    self.W1     = tf.keras.layers.Dense(units)
    self.W2     = tf.keras.layers.Dense(units)
    self.V      = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    attention_hidden_layer = (tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis)))

    score = self.V(attention_hidden_layer)

    attention_weights = tf.nn.softmax(score, axis=1)

    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class Encoder(tf.keras.Model):
  def __init__(self, embedding_dim, verbose = True):
    super(Encoder, self).__init__()

    self.verbose = verbose
    self.fc = tf.keras.layers.Dense(embedding_dim)

  def call(self, x):
    x = self.fc(x)
    x = tf.nn.relu(x)
    return x

class Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size, verbose = True):
    super(Decoder, self).__init__()

    self.verbose       = verbose
    self.units         = units
    self.vocab_size    = vocab_size
    self.embedding     = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.lstm          = tf.keras.layers.LSTM(self.units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
    self.fc1           = tf.keras.layers.Dense(self.units)

    self.dropout = tf.keras.layers.Dropout(0.5, noise_shape=None, seed=None)
    self.batchnormalization = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                               beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones',
                               beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)


    self.fc2       = tf.keras.layers.Dense(self.vocab_size)
    self.attention = BahdanauAttention(self.units)

  def call(self, x, features, hidden):
    context_vector, attention_weights = self.attention(features, hidden)

    x = self.embedding(x)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    output, mem_state, carry_state = self.lstm(x)

    x = self.fc1(output)
    x = tf.reshape(x, (-1, x.shape[2]))
    x = self.dropout(x)
    x = self.batchnormalization(x)
    x = self.fc2(x)

    return x, mem_state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))


class CNN (tf.keras.Model):
  def __init__(self, config, option, data, verbose = True):
    super().__init__()

    print (config)
    self.config    = config
    self.option    = option
    self.data      = data

    self.embedding_dim = config['embedding_dim']
    self.units         = config['units']
    self.vocab_size    = data['vocab_size']
    self.tokenizer     = load(open(option['tokenizer'], 'rb'))

    print (self.embedding_dim, self.units, self.vocab_size)

    self.decoder   = Decoder(self.embedding_dim, self.units, self.vocab_size)
    self.encoder   = Encoder(self.embedding_dim)

    print ('decoder:', type(self.decoder))
    print ('encoder:', type(self.encoder))

    print (self.is_keras_or_tf_model(self.decoder))

    self.verbose   = verbose

    for dir in option['dirs']:
      if not path.exists(dir):
        makedirs(dir, exist_ok=True)

    model = InceptionV3(include_top=False, weights='imagenet')
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

    #if self.verbose:
    #    print (model.summary())

    self.load_data ()
    self.model = model;

  def is_keras_or_tf_model(self, obj):
    return isinstance(obj, (tf.keras.Model, tf.estimator.Estimator))

  def load_image(self, image_path):
    full_path = self.option['images_dir'] +'/' + image_path + '.jpg'
    img = tf.io.read_file(full_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.vgg16.preprocess_input(img)
    return img, image_path

  def map_func(self, img_name, cap):
    img_tensor = np.load(self.option['features'] + '/' + img_name.decode('utf-8') + '.jpg.npy')
    return img_tensor, cap

  def create_dataset(self, img_name, cap):
    dataset = tf.data.Dataset.from_tensor_slices((img_name, cap))

    dataset = dataset.map(lambda item1, item2: tf.numpy_function(
            self.map_func, [item1, item2], [tf.float32, tf.int32]),
            num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.shuffle(self.config['buffer_size']).batch(self.config['batch_size'])
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

  def loss_function(self, real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = self.loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

  @tf.function
  def train_step(self, img_tensor, target):
    loss = 0
    tokenizer = self.tokenizer
    hidden = self.decoder.reset_state(batch_size=target.shape[0])
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

    with tf.GradientTape() as tape:
      features = self.encoder(img_tensor)

      for i in range(1, target.shape[1]):
        predictions, hidden,_ = self.decoder(dec_input, features, hidden)
        loss += self.loss_function(target[:, i], predictions)
        dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))
    trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, trainable_variables))
    return loss, total_loss

  @tf.function 
  def val_step(self, img_tensor, target):
    loss = 0
    tokenizer = self.tokenizer
    hidden     = self.decoder.reset_state(batch_size=target.shape[0])
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)
    features   = self.encoder(img_tensor)

    for i in range(1, target.shape[1]):
      predictions, hidden, _ = self.decoder(dec_input, features, hidden)
      loss += self.loss_function(target[:, i], predictions)
      dec_input = tf.expand_dims(target[:, i], 1)

    avg_loss = (loss / int(target.shape[1]))
    return loss, avg_loss

  def load_data (self):
    img_name_vector = self.data['dataset_train'] | self.data['dataset_valid'] | self.data['dataset_test']
    encode_images = sorted(set(img_name_vector))
    image_dataset = tf.data.Dataset.from_tensor_slices(encode_images)
    image_dataset = image_dataset.map(self.load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(64)

#    for img, path in tqdm(image_dataset, position=0, leave=True):
#      batch_features = self.model(img)
#      batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))
#
#      for bf, p in zip(batch_features, path):
#        path_of_feature = self.option['features'] + '/' + p.numpy().decode("utf-8") + '.jpg'
#        np.save(path_of_feature, bf.numpy())

  def Fit (self):
    train_dataset       = self.create_dataset(self.data['img_name_train'], self.data['caption_train'])
    valid_dataset       = self.create_dataset(self.data['img_name_valid'], self.data['caption_valid'])
    self.optimizer      = tf.keras.optimizers.Adam()
    self.loss_object    = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    loss_plot     = []
    val_loss_plot = []
    best_val_loss = float("inf")
    start_epoch   = 0
    start         = time.time()
    num_steps_train = len(self.data['img_name_train'])
    num_steps_val   = len(self.data['img_name_valid'])

    for epoch in range(start_epoch, self.config['epochs']):
      start = time.time()
      total_loss       = 0
      total_val_loss   = 0

      for (batch, (img_tensor, target)) in tqdm(enumerate(train_dataset), position=0, leave=True):
        batch_loss, t_loss = self.train_step(img_tensor, target)
        total_loss += t_loss

        if batch % 100 == 0:
          loss = batch_loss.numpy() / int(target.shape[1])
            
          print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, loss))            
                    
      for (batch_val, (img_tensor_val, target_val)) in enumerate(valid_dataset):
        batch_val_loss, t_val_loss = self.val_step(img_tensor_val, target_val)
        total_val_loss += t_val_loss

      loss_mean = total_loss / num_steps_train
      loss_plot.append(loss_mean)
      val_loss_mean = total_val_loss / num_steps_val
      val_loss_plot.append(val_loss_mean)
    
      print('Epoch {} Loss {:.6f} Val Loss {:.6f}'.format(epoch + 1, loss_mean, val_loss_mean))

      if val_loss_mean < best_val_loss:
        print('val_loss improved from %.4f to %.4f' % (best_val_loss, val_loss_mean))
        best_val_loss = val_loss_mean
        self.encoder.save_weights(self.option['encoder_model'])
        self.decoder.save_weights(self.option['decoder_model'])
      else:
        print('val_loss did not improve from %.4f' % (best_val_loss))

    if utils.is_notebook():
      import matplotlib.pyplot as plt
    else:
      import plotext as plt

    plt.plot(loss_plot, label='loss')
    plt.plot(val_loss_plot, label='val_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    plt.show()


  def generate_desc (self, image):
    attention_plot = np.zeros((self.data['max_length'], self.config['attention_features_shape']))
    hidden = self.decoder.reset_state(batch_size=1)

    temp_input      = tf.expand_dims(self.load_image(image)[0], 0)
    img_tensor_val  = self.model(temp_input)
    img_tensor_val  = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features  = self.encoder(img_tensor_val)
    dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']], 0)
    result = ['<start>']

    for i in range(self.data['max_length']):
        predictions, hidden, attention_weights = self.decoder(dec_input, features, hidden)
        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()
        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(self.tokenizer.index_word[predicted_id])

        if self.tokenizer.index_word[predicted_id] == '<end>':
            break

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    result = ' '.join(result)
    return result, attention_plot

  def calculate_scores(self, actual, predicted):
    smooth  = SmoothingFunction().method4
    bleu1   = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0),            smoothing_function=smooth)*100
    bleu2   = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0),          smoothing_function=smooth)*100
    bleu3   = corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0),        smoothing_function=smooth)*100
    bleu4   = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25),  smoothing_function=smooth)*100
    print('BLEU-1: %f' % bleu1)
    print('BLEU-2: %f' % bleu2)
    print('BLEU-3: %f' % bleu3)
    print('BLEU-4: %f' % bleu4)

  def evaluate_model(self):
    actual, predicted = list(), list()
    for key, desc_list in tqdm(self.data['test_descriptions'].items(), position=0, leave=True):
        yhat, _ = self.generate_desc(key)
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())
    print('Sampling:')
    self.calculate_scores(actual, predicted)


  def load_dec_enc_weight (self):
    self.encoder.built = True
    self.decoder.build = True
    self.encoder.load_weights (self.option['encoder_model'])
    self.decoder.load_weights (self.option['decoder_model'])

  def Evaluate (self):
    self.evaluate_model ()

  def plot_attention(self, image, result, attention_plot):
    temp_image = np.array(Image.open(image))
    fig = plt.figure(figsize=(10, 10))
    len_result = len(result)-1
    print('len_result: ', len_result)
    if len_result == 5:
      len_reult = 4

    for l in range(len_result):
      temp_att = np.resize(attention_plot[l], (8, 8))
      ax = fig.add_subplot(len_result//2, len_result//2, l+1)
      ax.set_title(result[l])
      img = ax.imshow(temp_image)
      ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()

  def clean_caption(self, caption):
    caption_list = caption.split()
    caption_list = caption_list[1:len(caption_list)-1]
    
    if '<unk>' in caption_list:
        caption_list = caption_list.remove('<unk>')
    if caption_list == None:
        caption = None
    elif len(caption_list) > 1:
        caption = ' '.join(caption_list)
    elif len(caption_list) == 1:
        caption = caption_list[0]
    else:
        caption = ''
    return caption

  def generate_captions(self, count):
    c = 0
    for key, desc_list in self.data['test_descriptions'].items():
      filename = self.option['images_dir'] + '/' + key + '.jpg'
      display(Image.open(filename))
      for i, desc in enumerate(desc_list):
        print('Original ' + str(i+1) + ': ' + self.clean_caption(desc_list[i]))
      desc, attention_plot = self.generate_desc(key)
      references = [d.split() for d in desc_list]
      smooth = SmoothingFunction().method4
      desc_bleu = sentence_bleu(references, desc.split(), weights=(1.0, 0, 0, 0), smoothing_function=smooth)*100
      captions = self.clean_caption(desc)
      print('Sampling (BLEU-1: %f): %s' % (desc_bleu, captions))
      if ((desc_bleu > 20.0) and captions != None and (len(captions.split(' ')) > 2)):
        if utils.is_notebook():
        	self.plot_attention(filename, desc.split(), attention_plot)
      c += 1
      if c == count:
        break

  def Generate (self, count):
    self.load_dec_enc_weight ()
    self.generate_captions (count)

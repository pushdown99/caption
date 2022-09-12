import tensorflow as tf

class RNN(tf.keras.Model):
  def __init__(self, config, params, attention, verbose = True):
    super().__init__()

    self.config  = config
    self.params  = params
    self.verbose = verbose
    self.units         = config['units']
    self.vocab_size    = params['vocab_size']
    self.embedding_dim = config['embedding_dim']

    self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)
    self.lstm = tf.keras.layers.LSTM(self.units,
                  return_sequences=True,
                  return_state=True,
                  recurrent_initializer='glorot_uniform')
    self.fc1 = tf.keras.layers.Dense(self.units)

    self.dropout = tf.keras.layers.Dropout(0.5, noise_shape=None, seed=None)
    self.batchnormalization = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                               beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones',
                               beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)


    self.fc2       = tf.keras.layers.Dense(self.vocab_size)

    self.attention = attention

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


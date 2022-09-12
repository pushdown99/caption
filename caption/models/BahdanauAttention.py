import tensorflow as tf

class Attention (tf.keras.Model):
  def __init__(self, data):
    super().__init__()

    self.data   = data
    self.W1     = tf.keras.layers.Dense(data['units'])
    self.W2     = tf.keras.layers.Dense(data['units'])
    self.V      = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    attention_hidden_layer = (tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis)))

    score = self.V(attention_hidden_layer)

    attention_weights = tf.nn.softmax(score, axis=1)

    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

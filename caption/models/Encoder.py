import tensorflow as tf

class CNN(tf.keras.Model):
  def __init__(self, config, verbose = True):
    super().__init__()

    self.config  = config
    self.verbose = verbose
    self.fc = tf.keras.layers.Dense(config['embedding_dim'])

  def call(self, x):
    x = self.fc(x)
    x = tf.nn.relu(x)
    return x


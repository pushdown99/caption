from os import path
from os import makedirs
from os import listdir
from os.path import isfile
from pickle import dump
from tensorflow.keras import layers
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model

class cnn:
  def __init__(self, dir = "datasets/Flickr8k/Flicker8k_Dataset", verbose = True):
    if not path.exists(dir):
      raise FileNotFoundError("Dataset directory does not exist: %s" % dir)

    self.dir     = dir
    self.out     = 'files/efficientnetb0'
    self.verbose = verbose

    if not path.exists(self.out):
      makedirs(self.out, exist_ok=True)

    model = EfficientNetB0(include_top=False, weights='imagenet')
    #model = EfficientNetB0(input_shape=(*(299, 299), 3), include_top=False, weights="imagenet",)
    #model.trainable = False
    #model_out = model.output
    #model_out = layers.Reshape((-1, 1280))(model_out)
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output) # model.layers[-1].output

    if self.verbose:
      print(model.summary())

    if not isfile(path.join(self.out, 'features.pkl')):
      features = dict()
      for name in listdir(dir):
        filename = dir + '/' + name
        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        image_id = name.split('.')[0]
        features[image_id] = feature

      if self.verbose:
        print('Extracted Features: %d' % len(features))

      dump(features, open(path.join(self.out, 'features.pkl'), 'wb'))



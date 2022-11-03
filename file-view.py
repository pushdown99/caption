from pickle     import dump,load
import itertools

filename = "files/features_coco2014_vgg16.pkl"

all_features = load(open(filename, 'rb'))

for k in all_features:
  print (k)

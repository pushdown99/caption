Config = {
  'embedding_dim':            256,
  'units':                    2,
  'batch_size':               64,
  'buffer_size':              1000,
  'features_shape':           512,
  'attention_features_shape': 25,
  'epochs':                   1,
}

Flickr8kOpts = {
  'datset_name':        'Flickr8k',
  'dirs':               ['files/Flickr8k'],
  'dataset_dir':        'datasets/Flickr8k/',
  'images_dir':         'datasets/Flickr8k/Flicker8k_Dataset',
  'images_mapping':     'datasets/Flickr8k/Flickr8k.token.txt',
  'images_train':       'datasets/Flickr8k/Flickr_8k.trainImages.txt',
  'images_valid':       'datasets/Flickr8k/Flickr_8k.devImages.txt',
  'images_test':        'datasets/Flickr8k/Flickr_8k.testImages.txt',
  'desciption_mapping': 'files/Flickr8k/descriptions_mapping.txt',
  'tokenizer':          'files/Flickr8k/tokenizer.pkl',
  'example_image':      'datasets/Flickr8k/Flicker8k_Dataset/667626_18933d713e.jpg',
  'example_id':         '667626_18933d713e',
}

InceptionV3Opts = {
  'model_name':       'InceptionV3',
  'dirs':             ['files/inceptionv3/features'],
  'images_dir':       'datasets/Flickr8k/Flicker8k_Dataset',
  'features':         'files/inceptionv3/features',
  'encoder_model':    'files/inceptionv3/encoder.h5',
  'decoder_model':    'files/inceptionv3/decoder.h5',
}


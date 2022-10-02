config = {
  'embedding_dim':            256,
  'units':                    256,
  'num_of_epochs':            50,
  'batch_size':               32,
}

flickr8k_opts = {
  'dataset_name':       'flickr8k',
  'dirs':              ['files', 'models'],
  'images_dir':         'datasets/Flickr8k/Flicker8k_Dataset',
  'images_mapping':     'datasets/Flickr8k/Flickr8k.token.txt',
  'images_train':       'datasets/Flickr8k/Flickr_8k.trainImages.txt',
  'images_valid':       'datasets/Flickr8k/Flickr_8k.devImages.txt',
  'images_test':        'datasets/Flickr8k/Flickr_8k.testImages.txt',
  'limits_train':       0,
  'limits_valid':       0,
  'limits_test':        10,
  'example_image':      'datasets/Flickr8k/Flicker8k_Dataset/667626_18933d713e.jpg',
  'example_id':         '667626_18933d713e',
}

vgg16_opts = {
  'model_name':       'vgg16',
  'dirs':            ['files', 'models'],
  'image_size':       224,
  'input_size':       4096,
}

inceptionv3_opts = {
  'model_name':       'inceptionv3',
  'dirs':            ['files', 'models'],
  'image_size':       299,
  'input_size':       2048,
}

efficientnet_opt = {
  'model_name':       'efficientnet',
  'dirs':            ['files', 'models'],
  'image_size':       224,
  'input_size':       1000,
}

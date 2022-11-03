config = {
  'embedding_dim':            256,
  'units':                    256,
  'num_of_epochs':            50,
  'batch_size':               32,
}

#Data = {
#  'dataset_train':      self.dataset_train,
#  'dataset_valid':      self.dataset_valid,
#  'dataset_test':       self.dataset_test,
#  'img_name_train':     self.img_name_train,
#  'img_name_valid':     self.img_name_valid,
#  'img_name_test':      self.img_name_test,
#  'train_descriptions': self.train_descriptions,
#  'valid_descriptions': self.valid_descriptions,
#  'test_descriptions':  self.test_descriptions,
#  'train_features':     self.train_features,
#  'valid_features':     self.valid_features,
#  'test_features':      self.test_features,
#  'vocab_size':         self.vocab_size,
#  'max_length':         self.max_length,
#  'tokenizer':          self.tokenizer,
#}

flickr8k_opts = {
  'dataset_name':       'flickr8k',
  'dirs':              ['files', 'models'],
  'images_dir':        ['datasets/Flickr8k/Flicker8k_Dataset'],
  'tests_dir':          'datasets/Flickr8k/Flicker8k_Dataset',
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

coco2014_opts = {
  'dataset_name':       'coco2014',
  'dirs':              ['files', 'models'],
  'dataset_dir':        'datasets/COCO',
  'images_dir':        ['datasets/COCO/train2014', 'datasets/COCO/val2014'],
  'tests_dir':          'datasets/COCO/val2014',
  'caption_train':      'datasets/COCO/annotations/captions_train2014.json',
  'caption_valid':      'datasets/COCO/annotations/captions_val2014.json',
  'limits_train':       0,
  'limits_valid':       0,
  'limits_test':        10,
  'example_image':      'datasets/COCO/train2014/COCO_train2014_000000247789.jpg',
  'example_id':         'COCO_train2014_000000247789',
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

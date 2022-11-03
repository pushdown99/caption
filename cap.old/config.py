parameters = {
}

coco2014_cfg = {
  'name':           'coco2014',
  'dataset_dir':    'datasets/COCO',
  'images_dir':     'datasets/COCO/train2014',
  'train_file':     'datasets/COCO/annotations/captions_train2014.json',
  'valid_file':     'datasets/COCO/annotations/captions_val2014.json',
  'train_limit':    3000000,
  'valid_limit':    10000,
  'val_test_split': 0.8,
  'example_image':  'datasets/COCO/train2014/COCO_train2014_000000247789.jpg',
  'example_id':     'COCO_train2014_000000247789',
}

flickr8k_cfg = {
  'name':           'flickr8k',
  'dataset_dir':    'datasets/Flickr8k',
  'images_dir':     'datasets/Flickr8k/Flicker8k_Dataset',
  'caption_file':   'datasets/Flickr8k/Flickr8k.token.txt',
  'train_file':     'datasets/Flickr8k/Flickr_8k.trainImages.txt',
  'valid_file':     'datasets/Flickr8k/Flickr_8k.devImages.txt',
  'train_limit':    3000000,
  'valid_limit':    10000,
  'val_test_split': 1.0,
  'example_image':  'datasets/Flickr8k/Flicker8k_Dataset/667626_18933d713e.jpg',
  'example_id':     '667626_18933d713e',
}

vgg16_cfg = {
  'name':                       'vgg16',

  'verbose':                    True,
  'image_size':                 224,
  'num_of_epochs':              10,
  'batch_size':                 64,
  'buffer_size':                1000,
  'embedding_dim':              256,
  'units':                      512,
  'top_k':                      5000,
  'features_shape':             512,
  'attention_features_shape':   64,
}

inceptionv3_cfg = {
  'name':                       'inceptionv3',

  'verbose':                    True,
  'image_size':                 299,
  'num_of_epochs':              10,
  'batch_size':                 64,
  'buffer_size':                1000,
  'embedding_dim':              256,
  'units':                      512,
  'top_k':                      5000,
  'features_shape':             512,
  'attention_features_shape':   64,
}

efficientb0_cfg = {
  'name':                       'efficientb0',

  'verbose':                    True,
  'image_size':                 299,
  'num_of_epochs':              10,
  'batch_size':                 64,
  'buffer_size':                1000,
  'embedding_dim':              160,
  'units':                      512,
  'top_k':                      5000,
  'features_shape':             512,
  'attention_features_shape':   64,
}


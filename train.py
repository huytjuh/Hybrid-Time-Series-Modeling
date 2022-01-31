import time
import pandas as pd

from options.train_options import TrainOptions


if __name__ == '__main__':
  opt = TrainOptions().parse()          # training options
  dataset = create_dataset(opt)         # initialize and preprocess dataset
  dataset_size = len(dataset)           
  print('Size: %d' % dataset_size)
  
  model = create_model(opt)             # create a model given opt.model and other options
  model.setup(opt)
  

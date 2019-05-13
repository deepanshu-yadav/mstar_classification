
from src.pre_trained_scripts.vgg16 import Vgg16
# from __future__ import division,print_function

import os, json
from glob import glob
import numpy as np
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt

import src.pre_trained_scripts.utils
# ; reload(utils)
from src.pre_trained_scripts.utils import plots


vgg = Vgg16()
batch_size= 4
path = os.path.join( os.getcwd() ,  'datasets' , 'dogscats' , 'sample' )

batches = vgg.get_batches( os.path.join(path ,'train' ), batch_size=batch_size)


val_batches = vgg.get_batches(   os.path.join(path ,'valid' ), batch_size=batch_size*2)
vgg.finetune(batches)
vgg.fit(batches, val_batches, nb_epoch=1)




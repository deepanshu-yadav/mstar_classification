import os
import src.pre_processing.get_data as get
import numpy as np
import h5py
from src.tensor_utilities.models import *
from src.tensor_utilities.training import *

BASE_PATH = os.path.join(os.getcwd(), 'datasets', 'SARBake')
img_size = 128
num_channels = 3
split_factor = 0.2
X_full , cls ,labels  ,class_names = get.get_datset_for_imbalanced_classes(BASE_PATH, img_size, img_size, num_channels)

name_of_experiment = '5_layer'
logs_path = "logs/5_layer/"
learning_rate = 0.001
no_of_epochs = 1500
batch_size = 100

print('dataset loaded...')

#idx = np.random.permutation(X_full.shape[0])
idx = np.random.permutation(X_full.shape[0])

test_idx = int(X_full.shape[0]*split_factor)
Y = X_full[idx[:test_idx]]
X = X_full[ idx[ test_idx:X_full.shape[0] ] ]

cls_test = cls[idx[:test_idx]]
cls_train = cls[ idx[ test_idx:X_full.shape[0] ] ]

labels_test = labels[idx[:test_idx]]
labels_train = labels[ idx[ test_idx:X_full.shape[0] ] ]
num_classes = len(class_names)



print( labels.shape )

X = X / X.max()
Y = Y / Y.max()


h5f1 = h5py.File('mstar.h5' , 'w')
h5f1.create_dataset('train' ,data= X)

#h5f2 = h5py.File('test_mstar.h5' , 'w')
h5f1.create_dataset('test' ,data= Y)

#h5f3 = h5py.File('cls_train.h5' , 'w')
h5f1.create_dataset('cls_train' ,data= cls_train)

#h5f4 = h5py.File('cls_test.h5' , 'w')
h5f1.create_dataset('cls_test' ,data= cls_test)

#h5f5 = h5py.File('labels_train.h5' , 'w')
h5f1.create_dataset('labels_train' ,data= labels_train)

#h5f6 = h5py.File('labels_test.h5' , 'w')
h5f1.create_dataset('labels_test' ,data= labels_test)

#h5f7 = h5py.File('class_names.h5' , 'w')
#h5f1.create_dataset('class_names' ,data= class_names)


# h5r1 = h5py.File('train_mstar.h5' , 'r')
# b = h5r1['train_dataset'][:]


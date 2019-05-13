import os
import src.pre_processing.get_data as get
import numpy as np

from src.tensor_utilities.models import *
from src.tensor_utilities.training import *

BASE_PATH = os.path.join(os.getcwd(), 'datasets', 'SARBake')
img_size = 128
num_channels = 3
split_factor = 0.2
X_full , cls ,labels  ,class_names= get.get_datset_for_imbalanced_classes(BASE_PATH,img_size,img_size,num_channels)
name_of_experiment = 'simple_cnn_with_drop_out'
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

print( labels.shape )

X =  X / X.max()
Y = ( Y) / Y.max()
num_classes = len(class_names)

model = Model()
X_full = None
with model.graph.as_default():
    x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    prob = tf.placeholder_with_default(1.0, shape=())
    # prob = True
    # prob_test = tf.placeholder_with_default(1 ,shape= ())
    # prob_test = False
    model.getsimpleCNNModel(x, num_classes, is_train=prob)
    # model.getFiveLayerModel(x, num_classes)

    model.getLoss(y_true)

    global_step = tf.Variable(initial_value=0,
                              name='global_step', trainable=False)

    model.getOptimizer(global_step=global_step, learning_rate=learning_rate)

    model.getAccuracy(y_true)

    train = Train(learning_rate=learning_rate, bath_size=batch_size, no_of_epochs=no_of_epochs, graph=model.graph,
                  accuracy=model.accuracy, loss=model.loss, name_of_experiment=name_of_experiment)

    train.saver.restore(sess=train.session, save_path=train.save_path)

    # train.optimize(x, y_true, global_step, model.optimizer, model.accuracy, X, labels_train,Y , labels_test , prob=prob )
    # train.optimize(x, y_true, global_step, model.optimizer, model.accuracy, X, labels_train,Y , labels_test ,class_imbalance=True ,logs_path=logs_path )

    train.print_test_accuracy(x=x,y_true=y_true,y_pred_cls=model.class_predicted,images_test=Y,labels_test=labels_test,cls_test=cls_test,class_names=class_names,  prob = prob , show_confusion_matrix=True,show_example_errors=True)
    # train.print_test_accuracy(x=x, y_true=y_true, y_pred_cls=model.class_predicted, images_test=Y,
    #                           labels_test=labels_test, cls_test=cls_test, class_names=class_names,
    #                           show_confusion_matrix=True, show_example_errors=True)

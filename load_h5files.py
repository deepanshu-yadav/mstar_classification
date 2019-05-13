import os
import src.pre_processing.get_data as get
import numpy as np
import h5py
from src.tensor_utilities.models import *
from src.tensor_utilities.training import *


h5r1 = h5py.File('mstar.h5' , 'r')
X = h5r1['train'][:]
print(type(X))
#h5r1 = h5py.File('test_mstar.h5' , 'r')
Y = h5r1['test'][:]

#h5r1 = h5py.File('cls_train.h5' , 'r')
cls_train = h5r1['cls_train'][:]

#h5r1 = h5py.File('cls_test.h5' , 'r')
cls_test = h5r1['cls_test'][:]

#h5r1 = h5py.File('labels_train.h5' , 'r')
labels_train = h5r1['labels_train'][:]

#h5r1 = h5py.File('train_mstar.h5' , 'r')
labels_test = h5r1['labels_test'][:]

#h5r1 = h5py.File('train_mstar.h5' , 'r')
#b = h5r1['train_dataset'][:]

BASE_PATH = os.path.join(os.getcwd(), 'datasets', 'SARBake')
class_names = get.get_class_names(BASE_PATH)


img_size = 128
num_channels = 3
split_factor = 0.2

name_of_experiment = 'simple_cnn_with_drop_out'
logs_path = "logs/simple_cnn/"
learning_rate = 0.001
no_of_epochs = 1500
batch_size = 100

num_classes = len(class_names)

model = Model()
X_full = None
with model.graph.as_default():
    x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    prob = tf.placeholder_with_default(0.5, shape=())
    # prob = True
    # prob_test = tf.placeholder_with_default(1 ,shape= ())
    # prob_test = False
    model.getsimpleCNNModel(x, num_classes, is_train=prob)
    #model.getFiveLayerModel(x, num_classes)
    #model.getFiveLayerModel(x, num_classes , is_train=prob)

    model.getLoss(y_true)

    global_step = tf.Variable(initial_value=0,
                              name='global_step', trainable=False)

    model.getOptimizer(global_step=global_step, learning_rate=learning_rate)

    model.getAccuracy(y_true)

    train = Train(learning_rate=learning_rate, bath_size=batch_size, no_of_epochs=no_of_epochs,
                  graph=model.graph, accuracy=model.accuracy, loss=model.loss,
                  name_of_experiment=name_of_experiment)

    train.saver.restore(sess=train.session, save_path=train.save_path)

    # train.optimize(x, y_true, global_step, model.optimizer,
    #                model.accuracy, X, labels_train,Y , labels_test , prob=prob )
    # train.optimize(x, y_true, global_step, model.optimizer, model.accuracy,
    #                X, labels_train,Y , labels_test ,class_imbalance=True ,logs_path=logs_path )
    # train.optimize(x, y_true, global_step, model.optimizer, model.accuracy,
    #                X, labels_train, Y, labels_test, class_imbalance=True, logs_path=logs_path,prob=prob )

    # train.print_test_accuracy(x=x,y_true=y_true,y_pred_cls=model.class_predicted,images_test=Y,labels_test=labels_test,cls_test=cls_test,class_names=class_names,  prob = prob , show_confusion_matrix=True,show_example_errors=True)
    train.print_test_accuracy(x=x, y_true=y_true, y_pred_cls=model.class_predicted, images_test=Y,
                              labels_test=labels_test, cls_test=cls_test, class_names=class_names,prob=prob,
                              show_confusion_matrix=True, show_example_errors=True)



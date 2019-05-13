from src.tensor_utilities.models import *
from src.tensor_utilities.training import *

import os
from src.pre_processing import get_data as g


def build_and_train():
    # os.chdir("../../")
    # img_size = 200
    # num_channels = 3
    # no_training = 600
    # no_test = 100
    # learning_rate = 0.00001
    # no_of_epochs = 500
    #
    # BASE_PATH = os.path.join(os.getcwd(), 'datasets', 'SARBake')
    # X, Y, cls_train, cls_test, labels_train, labels_test, class_names = g.get_datset_in_batch(BASE_PATH, img_size, img_size, no_training, no_test)
    # num_classes = len(class_names)
    #
    # model = Model()
    #
    # with model.graph.as_default():
    #     x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
    #     y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    #     #y_true_cls = tf.argmax(y_true, dimension=1)
    #     model.getsimpleCNNModel(x, num_classes)
    #
    #     model.getLoss(y_true)
    #
    #     global_step = tf.Variable(initial_value=0,
    #                               name='global_step', trainable=False)
    #
    #     model.getOptimizer(global_step=global_step, learning_rate=learning_rate)
    #
    #     model.getAccuracy(y_true=y_true)
    #
    #     # y_true_cls = tf.argmax(y_true, dimension=1)
    #     #
    #     # y_pred_cls = tf.argmax(model.pred, dimension=1)
    #     #
    #     # correct_prediction = tf.equal(  y_pred_cls , y_true_cls)
    #     #
    #     # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #
    #     train = Train(learning_rate=learning_rate, bath_size=32, no_of_epochs=no_of_epochs, graph=model.graph,accuracy=model.accuracy ,loss=model.loss)
    #
    #     train.optimize(x, y_true, global_step, model.optimizer, model.accuracy, X, labels_train, if_restore=True)
    #
    #     print("this is pre function"  ,cls_test  )
    #     print()
    #
    #     train.print_test_accuracy(x=x,y_true=y_true,y_pred_cls=model.class_predicted,images_test=Y,labels_test=labels_test,cls_test=cls_test,class_names=class_names,show_confusion_matrix=True)
    # os.chdir("../../")
    img_size = 100
    num_channels = 3
    no_training = 50
    no_test = 10
    learning_rate = 0.00001
    no_of_epochs = 2

    BASE_PATH = os.path.join(os.getcwd(), 'datasets', 'SARBake')
    X, Y, cls_train, cls_test, labels_train, labels_test, class_names = g.get_datset_in_batch(BASE_PATH, img_size,
                                                                                              img_size, no_training,
                                                                                              no_test)
    num_classes = len(class_names)

    model = Model()

    with model.graph.as_default():
        x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
        y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
        model.getsimpleCNNModel(x, num_classes)

        model.getLoss(y_true)

        global_step = tf.Variable(initial_value=0,
                                  name='global_step', trainable=False)

        model.getOptimizer(global_step=global_step, learning_rate=learning_rate)

        model.getAccuracy(y_true)

        train = Train(learning_rate=learning_rate, bath_size=80, no_of_epochs=no_of_epochs, graph=model.graph,
                      accuracy=model.accuracy, loss=model.loss)

        # train.saver.restore(sess=train.session,save_path=train.save_path)

        train.optimize(x, y_true, global_step, model.optimizer, model.accuracy, X, labels_train, Y, labels_test)

        train.print_test_accuracy(x=x, y_true=y_true, y_pred_cls=model.class_predicted, images_test=Y,
                                  labels_test=labels_test, cls_test=cls_test, class_names=class_names,
                                  show_confusion_matrix=True, show_example_errors=True)


if __name__ == '__main__':
    build_and_train()

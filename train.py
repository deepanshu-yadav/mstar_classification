from src.tensor_utilities.models import *
from src.tensor_utilities.training import *
import os
from src.pre_processing import get_data as g


def build_restore_train():
    #os.chdir("../../")
    img_size = 128
    num_channels = 3
    no_training = 300
    no_test = 100
    learning_rate = 0.001
    no_of_epochs =  1500
    batch_size =    100
    BASE_PATH = os.path.join(os.getcwd(), 'datasets', 'SARBake')
    X, Y, cls_train, cls_test, labels_train, labels_test, class_names = g.get_datset_in_batch(BASE_PATH, img_size, img_size, no_training, no_test)
    X =  X / X.max()
    Y = ( Y) / Y.max()
    num_classes = len(class_names)

    model = Model()

    with model.graph.as_default():
        x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
        y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
        prob = tf.placeholder_with_default(0.5 ,shape= ())
        #prob= True
        #prob_test = tf.placeholder_with_default(1 ,shape= ())
        #prob_test = False
        #model.getsimpleCNNModel(x, num_classes,is_train=prob)
        #model.getsimpleCNNModel(x, num_classes )
        model.getFiveLayerModel(x, num_classes,is_train=prob)
        model.getLoss(y_true)

        global_step = tf.Variable(initial_value=0,
                                  name='global_step', trainable=False)

        model.getOptimizer(global_step=global_step, learning_rate=learning_rate)

        model.getAccuracy(y_true)
       
        train = Train(learning_rate=learning_rate, bath_size=batch_size, no_of_epochs=no_of_epochs, graph=model.graph,accuracy=model.accuracy ,loss=model.loss)
        
        train.saver.restore(sess=train.session,save_path=train.save_path)
        
        #train.optimize(x, y_true, global_step, model.optimizer, model.accuracy, X, labels_train,Y , labels_test , prob=prob )
        #train.optimize(x, y_true, global_step, model.optimizer, model.accuracy, X, labels_train,Y , labels_test  )

        train.print_test_accuracy(x=x,y_true=y_true,y_pred_cls=model.class_predicted,images_test=Y,labels_test=labels_test,cls_test=cls_test,class_names=class_names,  prob = prob , show_confusion_matrix=True,show_example_errors=True)
        #train.print_test_accuracy(x=x,y_true=y_true,y_pred_cls=model.class_predicted,images_test=Y,labels_test=labels_test,cls_test=cls_test,class_names=class_names,   show_confusion_matrix=True,show_example_errors=True)

if __name__=='__main__':
    build_restore_train()

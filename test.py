from src.tensor_utilities.models import *
from src.tensor_utilities.training import *

import os
from src.pre_processing import get_data as g
import src.tensor_utilities.testing as tests


#C:\Users\Dell\PycharmProjects\my_dev_env\pre_trained_models\inception

#BASE_PATH = os.path.join(os.getcwd() ,'pre_trained_models' , 'inception')
#BASE_PATH = os.path.join(os.getcwd() ,'pre_trained_models' , 'ssd_mobilenet_v1_coco_2018_01_28')
#BASE_PATH = os.path.join(os.getcwd() ,'pre_trained_models' , 'inception' , 'classify_image_graph_def.pb')
#print(BASE_PATH)
from src.plotting import plotter as ptr
def build_restore_test():
    # os.chdir("../../")
    img_size = 100
    num_channels = 3
    no_training = 1
    no_test = 1
    learning_rate = 0.00001
    no_of_epochs = 2

    names = [  '2S1' ,'BMP2' , 'BRDM2' , 'BTR60' ,'BTR70', 'D7' ,  'T62' , 'T72', 'ZIL131' ,'ZSU23' ]

    BASE_PATH = 'test.png'
    Y = g.get_single_image(BASE_PATH, img_size,img_size)

    print(Y.std() )
    Y = ( Y  ) /  Y.max()

    print(Y.min() , Y.max() , Y.mean()  , Y.std() )
    num_classes = 10

    model = Model()

    with model.graph.as_default():
        x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
        y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
        #model.getPreDefinedModel(BASE_PATH)
        model.getsimpleCNNModel(x, num_classes)

        model.getLoss(y_true)

        global_step = tf.Variable(initial_value=0,
                                  name='global_step', trainable=False)

        model.getOptimizer(global_step=global_step, learning_rate=learning_rate)

        model.getAccuracy(y_true)

        train = Train(learning_rate=learning_rate, bath_size=100, no_of_epochs=no_of_epochs, graph=model.graph,
                      accuracy=model.accuracy, loss=model.loss)
        train.saver.restore(sess=train.session, save_path=train.save_path)
        print(  "class clssified  is %s"  % names[tests.classify(x,Y, model=model ,session=train.session)] )

        #print(train.print_layer_names())

        grads = train.get_gradients_any_layer(x,Y,name_of_layer="layer_conv1/convolution:0")
        gdr = np.array(grads)
        # layer_conv1
        ptr.plot_montage(gdr)

if __name__ == '__main__':
    build_restore_test()

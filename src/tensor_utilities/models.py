import tensorflow as tf

import os
class Model:
    def __init__(self):
        self.pred=None
        self.loss=None
        self.optimizer=None
        self.logits=None
        self.pred_max=None
        self.graph=tf.Graph()
        self.accuracy=None
        self.class_predicted=None
    def getsimpleCNNModel(self,x,num_classes,is_train = None):
        with self.graph.as_default():
            net = x

            net = tf.layers.conv2d(inputs=net, name='layer_conv1', padding='same',
                                   filters=16, kernel_size=5, activation=tf.nn.relu)
            # if is_train != None:
            #     net = tf.layers.dropout(net, training=is_train)
            layer_conv1 = net

            net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
            net = tf.layers.batch_normalization(net)

            net = tf.layers.conv2d(inputs=net, name='layer_conv2', padding='same',
                                   filters=36, kernel_size=5, activation=tf.nn.relu)
            # if is_train != None:
            #     net = tf.layers.dropout(net, training=is_train)
            layer_conv2 = net

            net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
            net = tf.layers.batch_normalization(net)

            net = tf.contrib.layers.flatten(net)
            net = tf.layers.batch_normalization(net)
            net = tf.layers.dense(inputs=net, name='layer_fc1',
                                  units=128, activation=tf.nn.relu)

            if is_train != None:
                net = tf.layers.dropout(net, is_train)

            net = tf.layers.dense(inputs=net, name='layer_fc_out',
                                  units=num_classes, activation=None)

            logits = net

            y_pred = tf.nn.softmax(logits=logits)

            self.pred = y_pred

            self.logits=logits

            self.class_predicted=tf.argmax(self.pred, dimension=1)
    def getCNNModel(self,x,num_classes):
        with self.graph.as_default():
            net = x

            net = tf.layers.conv2d(inputs=net, name='layer_conv1', padding='same',
                                   filters=16, kernel_size=3, activation=tf.nn.relu)
            net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

            net = tf.layers.conv2d(inputs=net, name='layer_conv2', padding='same',
                                   filters=32, kernel_size=5, activation=tf.nn.relu)
            net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

            net = tf.layers.conv2d(inputs=net, name='layer_conv3', padding='same',
                                   filters=64, kernel_size=5, activation=tf.nn.relu)
            net = tf.layers.average_pooling2d(inputs=net, pool_size=2, strides=2)


            net = tf.contrib.layers.flatten(net)

            net = tf.layers.dense(inputs=net, name='layer_fc1',
                                  units=128, activation=tf.nn.relu)
            # net = tf.layers.dense(inputs=net, name='layer_fc2',
            #                       units=64, activation=tf.nn.relu)
            net = tf.layers.dense(inputs=net, name='layer_fc_out',
                                  units=num_classes, activation=None)

            logits = net
            y_pred = tf.nn.softmax(logits=logits)
            self.pred = y_pred
            self.logits=logits
            self.class_predicted=tf.argmax(self.pred, dimension=1)

    def getFiveLayerModel(self,x,num_classes,is_train = None):
        with self.graph.as_default():
            net = x

            net = tf.layers.conv2d(inputs=net, name='layer_conv1',padding='valid',
                                   filters=48, kernel_size=5, activation=tf.nn.relu)
            net = tf.layers.average_pooling2d(inputs=net, pool_size=2, strides=2)
            net = tf.layers.batch_normalization(net)

            net = tf.layers.conv2d(inputs=net, name='layer_conv2', padding='valid',
                                   filters=96, kernel_size=5, activation=tf.nn.relu)

            net = tf.layers.average_pooling2d(inputs=net, pool_size=2, strides=2)
            net = tf.layers.batch_normalization(net)

            net = tf.layers.conv2d(inputs=net, name='layer_conv3', padding='valid',
                                   filters=128, kernel_size=3, activation=tf.nn.relu)

            net = tf.layers.average_pooling2d(inputs=net, pool_size=2, strides=2)
            net = tf.layers.batch_normalization(net)

            net = tf.layers.conv2d(inputs=net, name='layer_conv4', padding='valid',
                                   filters=128, kernel_size=3, activation=tf.nn.relu)

            net = tf.layers.average_pooling2d(inputs=net, pool_size=2, strides=2)
            net = tf.layers.batch_normalization(net)

            net = tf.layers.conv2d(inputs=net, name='layer_conv5', padding='valid',
                                   filters=256, kernel_size=3, activation=tf.nn.relu)

            net = tf.layers.average_pooling2d(inputs=net, pool_size=2, strides=2)
            net = tf.layers.batch_normalization(net)

            # net = tf.layers.conv2d(inputs=net, name='layer_conv3', padding='same',
            #                        filters=64, kernel_size=5, activation=tf.nn.relu)
            # net = tf.layers.average_pooling2d(inputs=net, pool_size=2, strides=2)


            net = tf.contrib.layers.flatten(net)
            net = tf.layers.batch_normalization(net)
            net = tf.layers.dense(inputs=net, name='layer_fc1',
                                  units=128, activation=tf.nn.relu)
            # net = tf.layers.dense(inputs=net, name='layer_fc2',
            #                       units=64, activation=tf.nn.relu)

            if is_train != None:
                net = tf.layers.dropout(net, is_train)
            net = tf.layers.dense(inputs=net, name='layer_fc_out',
                                  units=num_classes, activation=None)

            logits = net
            y_pred = tf.nn.softmax(logits=logits)
            self.pred = y_pred
            self.logits=logits
            self.class_predicted=tf.argmax(self.pred, dimension=1)

    def getPaperModel(self,x,num_classes):
        with self.graph.as_default():
            net = x

            net = tf.layers.conv2d(inputs=net, name='layer_conv1',padding='valid',
                                   filters=6, kernel_size=33, activation=tf.nn.relu)
            net = tf.layers.average_pooling2d(inputs=net, pool_size=2, strides=2)

            net = tf.layers.conv2d(inputs=net, name='layer_conv2', padding='valid',
                                   filters=12, kernel_size=13, activation=tf.nn.relu)

            net = tf.layers.average_pooling2d(inputs=net, pool_size=2, strides=2)

            # net = tf.layers.conv2d(inputs=net, name='layer_conv3', padding='same',
            #                        filters=64, kernel_size=5, activation=tf.nn.relu)
            # net = tf.layers.average_pooling2d(inputs=net, pool_size=2, strides=2)


            net = tf.contrib.layers.flatten(net)

            # net = tf.layers.dense(inputs=net, name='layer_fc1',
            #                       units=128, activation=tf.nn.relu)
            # net = tf.layers.dense(inputs=net, name='layer_fc2',
            #                       units=64, activation=tf.nn.relu)
            net = tf.layers.dense(inputs=net, name='layer_fc_out',
                                  units=num_classes, activation=None)

            logits = net
            y_pred = tf.nn.softmax(logits=logits)
            self.pred = y_pred
            self.logits=logits
            self.class_predicted=tf.argmax(self.pred, dimension=1)


    def getLoss(self,y_true,choice =1):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=self.logits)

        loss = tf.reduce_mean(cross_entropy)

        self.loss =loss

    def getOptimizer(self,global_step, learning_rate , choice=1):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss, global_step)
        self.optimizer=optimizer

    def getLogits(self):
        return self.logits
    def getAccuracy(self,y_true):
        y_true_cls = tf.argmax(y_true, dimension=1)

        correct_prediction = tf.equal(self.class_predicted, y_true_cls)

        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def getPreDefinedModel(self,path):
        with self.graph.as_default():

            # TensorFlow graphs are saved to disk as so-called Protocol Buffers
            # aka. proto-bufs which is a file-format that works on multiple
            # platforms. In this case it is saved as a binary file.

            # Open the graph-def file for binary reading.
            #path = os.path.join(path, path_graph_def)
            with tf.gfile.FastGFile(path, 'rb') as file:
                # The graph-def is a saved copy of a TensorFlow graph.
                # First we need to create an empty graph-def.
                graph_def = tf.GraphDef()

                # Then we load the proto-buf file into the graph-def.
                graph_def.ParseFromString(file.read())

                # Finally we import the graph-def to the default TensorFlow graph.
                tf.import_graph_def(graph_def, name='')







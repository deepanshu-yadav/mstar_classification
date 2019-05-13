import tensorflow as tf
import numpy as np
import time
import os
#from src.tensor_utilities.models import Model as model
import src.tensor_utilities.testing as ts
import src.plotting.plotter as plts
class Train:
    def __init__(self, learning_rate, bath_size, no_of_epochs, graph ,accuracy,loss,name_of_experiment='best_validation'):
        self.learning_rate = learning_rate
        self.bath_size = bath_size
        self.no_of_epochs = no_of_epochs
        self.graph = graph
        self.session = tf.Session(graph=graph)

        #self.summary_accuracy_train = tf.summary.scalar("accuracy_train", accuracy)
        self.summary_accuracy = tf.summary.scalar("accuracy", accuracy)
        self.summary_loss = tf.summary.scalar("loss", loss)
        #self.summary_loss_test = tf.summary.scalar("loss_test", loss)
        self.merged_summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        save_dir = 'checkpoints/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_path = os.path.join(save_dir,name_of_experiment )
        self.session.run(tf.global_variables_initializer())

    def random_batch(self,images_train,labels_train):
        # Number of images (transfer-values) in the training-set.
        num_images = len(labels_train)
        #print(" the length is  %d" %num_images)
        # Create a random index.
        idx = np.random.choice(num_images,
                               size = self.bath_size,
                               replace=False)



        # Use the random index to select random x and y-values.
        # We use the transfer-values instead of images as x-values.
        x_batch = images_train[idx]
        y_batch = labels_train[idx]

        return x_batch, y_batch
    def random_batch2(self,images_train,labels_train):
        # Number of images (transfer-values) in the training-set.
        num_images = len(labels_train)
        #print(" the length is  %d" %num_images)
        # Create a random index.
        idx = np.random.choice(num_images,
                               size = self.bath_size,
                               replace=False)

        n_cls = images_train.shape[1]



        # Use the random index to select random x and y-values.
        # We use the transfer-values instead of images as x-values.
        x_batch=np.zeros(shape=(self.bath_size , images_train.shape[2] ,images_train.shape[3]   ,images_train.shape[4] ))
        for i in range(self.bath_size):
            cls = int(idx[i] / n_cls)
            off = idx[i] % n_cls
            #print(idx[i] , cls,off)
            x_batch[  i, :, :, : ] = images_train[cls,off,:,:,:]
        #print(labels_train[idx])
        #print(idx)
        y_batch = labels_train[idx]

        return x_batch, y_batch
    def optimize(self,x,y_true,global_step,optimizer,accuracy, images_train, labels_train,images_test,labels_test,class_imbalance=False,prob=None ,logs_path="logs/" ):
        # Start-time used for printing time-usage below.
        #start_time = time.time()

        logs_path_train=logs_path+"train/"
        logs_path_test=logs_path+"test/"
        summary_writer_train = tf.summary.FileWriter(logs_path_train, graph=self.graph)
        summary_writer_test = tf.summary.FileWriter(logs_path_test, graph=self.graph)
        #print("  images shape is  " ,images_train.shape)
        for i in range(self.no_of_epochs):
            # Get a batch of training examples.
            # x_batch now holds a batch of images (transfer-values) and
            # y_true_batch are the true labels for those images.
            if not class_imbalance:
                x_batch, y_true_batch =self.random_batch2(images_train , labels_train)
            else:
                x_batch, y_true_batch = self.random_batch(images_train, labels_train)

            # Put the batch into a dict with the proper names
            # for placeholder variables in the TensorFlow graph.
            if prob !=None:
                feed_dict_train = {x: x_batch,
                                   y_true : y_true_batch,
                                   prob : 0.5}
                feed_dict_test = {x: images_test,
                                   y_true : labels_test,
                                  prob : 1}
            else:
                feed_dict_train = {x: x_batch,
                                   y_true: y_true_batch}
                feed_dict_test = {x: images_test,
                                  y_true: labels_test}

            # Run the optimizer using this batch of training data.
            # TensorFlow assigns the variables in feed_dict_train
            # to the placeholder variables and then runs the optimizer.
            # We also want to retrieve the global_step counter.
            i_global, _ = self.session.run([global_step, optimizer ],
                                      feed_dict=feed_dict_train)

            # Print status to screen every 100 iterations (and last).
            if (i_global % 10 == 0) or (i == self.no_of_epochs - 1):
                # Calculate the accuracy on the training-batch.
                batch_acc,summary_train_acc,summary_train_loss = self.session.run( [accuracy,self.summary_accuracy,self.summary_loss],
                                        feed_dict=feed_dict_train )
                summary_writer_train.add_summary(summary_train_acc, i_global  )
                summary_writer_train.add_summary(summary_train_loss, i_global  )

                msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
                print(msg.format(i_global, batch_acc))
                if (i_global % 50 == 0):
                    # val_acc,summary_test_acc,summary_test_loss = self.session.run([accuracy,self.summary_accuracy ,self.summary_loss],
                    #                   feed_dict=feed_dict_test)
                    # summary_writer_test.add_summary(summary_test_acc, i_global  )
                    # summary_writer_test.add_summary(summary_test_loss, i_global  )
                    # msg = "Global Step: {0:>6}, Validation  Accuracy: {1:>6.1%}"
                    # print(msg.format(i_global, val_acc))
                    self.saver.save(sess=self.session, save_path=self.save_path)
                    # Print status.




    def print_test_accuracy(self,x,y_true,y_pred_cls,images_test, labels_test, cls_test, class_names, prob=None ,test_batch_size=32 ,show_example_errors=False,
                            show_confusion_matrix=False):
        # Number of images in the test-set.
        num_test = images_test.shape[0]

        # Allocate an array for the predicted classes which
        # will be calculated in batches and filled into this array.
        cls_pred = np.zeros(shape=num_test, dtype=np.int)

        # Now calculate the predicted classes for the batches.
        # We will just iterate through all the batches.
        # There might be a more clever and Pythonic way of doing this.

        # The starting index for the next batch is denoted i.
        i = 0

        while i < num_test:
            # The ending index for the next batch is denoted j.
            j = min(i + test_batch_size, num_test)

            # Get the images from the test-set between index i and j.
            images = images_test[i:j, :]

            # Get the associated labels.
            labels = labels_test[i:j, :]

            # Create a feed-dict with these images and labels.
            if prob != None:
                feed_dict = {x: images,
                             y_true: labels,
                             prob : 1}
            else:
                feed_dict = {x: images,
                             y_true: labels}

            # Calculate the predicted class using TensorFlow.
            cls_pred[i:j] = self.session.run(y_pred_cls, feed_dict=feed_dict)

            # Set the start-index for the next batch to the
            # end-index of the current batch.
            i = j

        # Convenience variable for the true class-numbers of the test-set.
        cls_true = cls_test[0: num_test]
        #print(cls_true)
        #print(cls_pred)
        # print(cls_pred)

        # Create a boolean array whether each image is correctly classified.
        correct = (cls_true == cls_pred)

        # Calculate the number of correctly classified images.
        # When summing a boolean array, False means 0 and True means 1.
        correct_sum = sum(correct)
        # print(correct)
        # Classification accuracy is the number of correctly classified
        # images divided by the total number of images in the test-set.
        acc = float(correct_sum) / num_test
        print(num_test)
        # Print the accuracy.
        msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
        print(msg.format(acc, correct_sum, num_test))

        # Plot some examples of mis-classifications, if desired.
        if show_example_errors:
            print("Example errors:")
            #plts.plot_example_errors(  images_test = images_test ,correct=correct ,cls_true=cls_true ,class_names=class_names ,cls_pred=cls_pred)

        # Plot the confusion matrix, if desired.
        if show_confusion_matrix:
            print("Confusion Matrix:")
            ts.plot_confusion_matrix(cls_pred=cls_pred, cls_test=cls_test, class_names=class_names,
                                 num_classes=len(class_names), test_images=images_test)

    def create_feed_dict(self,x, image):
        feed_dict = {x: image}
        return feed_dict
    def get_gradients_any_layer(self, x, image, name_of_layer='conv/Conv2D:0'):
        # Create a feed-dict for the TensorFlow graph with the input image.
        feed_dict = self.create_feed_dict(x ,image=image)

        # Execute the TensorFlow session to get the predicted labels.
        # W = self.graph.get_tensor_by_name('conv_2/conv2d_params:0')
        feature = self.graph.get_tensor_by_name(name_of_layer)
        grad_list = []
        # first_layer = self.session.run(   W , feed_dict=feed_dict)

        # shape_of_tensor = tf.get_shape(feature)
        shape_of_tensor = tf.shape(feature).eval(session=self.session, feed_dict=feed_dict)
        #print(shape_of_tensor)
        for i in range(shape_of_tensor[-1]):
            #gradient = tf.gradients(feature[:, :, :, i], x)
            gradient = tf.gradients(tf.reduce_mean(feature[:,:,:,i]), x)
            grad_cal = self.session.run(gradient, feed_dict=feed_dict)
            grad_list.append(grad_cal)
        # print(len(grad_cal))
        return grad_list

    def print_layer_names(self):
        names = [op.name for op in self.graph.get_operations()]
        return  names

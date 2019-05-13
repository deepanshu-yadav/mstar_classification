# Import a function from sklearn to calculate the confusion-matrix.
from sklearn.metrics import confusion_matrix
import numpy as np
def plot_confusion_matrix(cls_pred,cls_test,class_names,num_classes,test_images):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_test[  0:test_images.shape[0]  ],  # True class for test-set.
                          y_pred=cls_pred)  # Predicted class.

    # Print the confusion matrix as text.
    for i in range(num_classes):
        # Append the class-name to each line.
        class_name = "({}) {}".format(i, class_names[i])
        print(cm[i, :], class_name)

    # Print the class-numbers for easy reference.
    class_numbers = ["({0})".format(i) for i in range(num_classes)]
    print("".join(class_numbers))


def create_feed_dict(x,image):
    feed_dict = {x: image}
    return feed_dict

# Split the test-set into smaller batches of this size.

def classify( x,image,model,session):


        # Create a feed-dict for the TensorFlow graph with the input image.
    feed_dict = create_feed_dict(x  , image)

        # Execute the TensorFlow session to get the predicted labels.
    pred = session.run(model.class_predicted, feed_dict=feed_dict)

        # Reduce the array to a single dimension.
    pred = np.squeeze(pred)
    return pred






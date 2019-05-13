from src.tensor_utilities.models import *
from src.tensor_utilities.training import *

import os
from src.pre_processing import get_data as g
import src.tensor_utilities.testing as tests
import src.tensor_utilities.NamesOfClasses  as noc

#BASE_PATH = os.path.join(os.getcwd() ,'pre_trained_models' , 'ssd_mobilenet_v1_coco_2018_01_28')
BASE_PATH = os.path.join(os.getcwd() ,'pre_trained_models' , 'inception' , 'classify_image_graph_def.pb')
DATA_DIR = os.path.join(os.getcwd() ,'pre_trained_models' , 'inception' )
path_uid_to_cls = "imagenet_2012_challenge_label_map_proto.pbtxt"
path_uid_to_name = "imagenet_synset_to_human_label_map.txt"

from src.plotting import plotter as ptr


def print_scores(name_lookup, pred, k=10, only_first_name=True):


    # Get a sorted index for the pred-array.
    idx = pred.argsort()

    # The index is sorted lowest-to-highest values. Take the last k.
    top_k = idx[-k:]

    # Iterate the top-k classes in reversed order (i.e. highest first).
    for cls in reversed(top_k):
        # Lookup the class-name.
        name = name_lookup.cls_to_name(cls=cls, only_first_name=only_first_name)

        # Predicted score (or probability) for this class.
        score = pred[cls]

        # Print the score and class-name.
        print("{0:>6.2%} : {1}".format(score, name))


def create_feed_dict(x, image):
    feed_dict = {x: image}
    return feed_dict

def classify_pretrained( x, image, output,sess):
    feed_dict = create_feed_dict(x, image)
    out=sess.run(output, feed_dict=feed_dict)
    out = np.squeeze(out)
    return out
def get_tensor_by_name(graph,name_of_tensor):
    return graph.get_tensor_by_name(name_of_tensor)

def build_restore_test():
    num_channels = 3
    img_size = 299
    # no_training = 300
    # no_test = 100
    # learning_rate = 0.001
    # no_of_epochs = 1500
    # batch_size = 100
    # BASE_PATH = os.path.join(os.getcwd(), 'datasets', 'SARBake')
    # X, Y, cls_train, cls_test, labels_train, labels_test, class_names = g.get_datset_in_batch(BASE_PATH, img_size,
    #                                                                                           img_size, no_training,
    #                                                                                           no_test)
    # X = X / X.max()
    # Y = (Y) / Y.max()
    # num_classes = len(class_names)


    BASE_PATH_IMAGE = 'sun.jpg'
    Y = g.get_single_image(BASE_PATH_IMAGE, img_size,img_size)
    Y=Y[0,:,:,:]
    tensor_name_softmax = "softmax:0"
    tensor_name_input_image = "DecodeJpeg:0"
    tensor_name_transfer_layer = "pool_3:0"

    name_lookup = noc.NameLookup(DATA_DIR ,path_uid_to_name ,path_uid_to_cls )

    model = Model()

    # with model.graph.as_default():
    #     model.getPreDefinedModel(BASE_PATH)
    #     x = get_tensor_by_name(model.graph, tensor_name_input_image)
    #     output_tensor = get_tensor_by_name(model.graph, tensor_name_softmax)
    # sess = tf.Session(graph=model.graph)
    # res = classify_pretrained(x, Y, output_tensor, sess)
    # print_scores(name_lookup, res, k=5, only_first_name=True)


    with model.graph.as_default():
        model.getPreDefinedModel(BASE_PATH)
        x = get_tensor_by_name(model.graph, tensor_name_input_image)
        output_tensor = get_tensor_by_name(model.graph, tensor_name_transfer_layer)
    sess = tf.Session(graph=model.graph)
    res = classify_pretrained(x, Y, output_tensor, sess)
    print(res.shape)
if __name__ == '__main__':
    build_restore_test()

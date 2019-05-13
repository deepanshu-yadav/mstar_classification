#
# import os
# from src.pre_processing import get_data as g
# from src.plotting import plotter as ptr
#
# os.chdir("../../")
# BASE_PATH = os.path.join( os.getcwd(),'datasets' ,'SARBake'  )
# X , Y, cls_train ,cls_test , labels_train , labels_test, class_names  =g.get_datset_in_batch(BASE_PATH,200,200,100,20)
# print(X.shape)
# print(class_names)
#
# images = X[0,0:9,:,:,:]
#
# # Get the true classes for those images.
# cls_true = cls_test[0:9]
#
# # Plot the images and labels using our helper-function above.
# #ptr.plot_images(images=images, cls_true=cls_true, class_names=class_names ,smooth=False)
#
# ptr.plot_montage(images)

from sklearn.datasets import fetch_20newsgroups


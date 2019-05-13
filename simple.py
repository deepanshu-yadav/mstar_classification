
import os
from src.pre_processing import get_data as g
BASE_PATH = os.path.join( os.getcwd(),'datasets' ,'SARBake')
X , Y, cls_train  ,cls_test , labels_train , labels_test, class_names  =g.get_full_dataset(BASE_PATH,200,200,500,100)
print(X.shape)
print(class_names)

import keras
from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.callbacks import TensorBoard
from sklearn.cluster import KMeans
from keras.applications.resnet50 import preprocess_input
import time
from sklearn import metrics
import glob as glob
import cv2
import numpy as np


NAME = 'board {}'.format(time.time())
tensorboard = TensorBoard(log_dir='../../photos/{}'.format(NAME))

num_classes = 2
resnet_weights_path = '../../resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))

# Say not to train first layer (ResNet) model. It is already trained
my_new_model.layers[0].trainable = False

def extract_vector(path):
    resnet_feature_list = []
    counter = 0
    for im in glob.glob(path):
        im = cv2.imread(im)
        im = cv2.resize(im,(200,200))
        img = preprocess_input(np.expand_dims(im.copy(), axis=0))
        resnet_feature = my_new_model.predict(img)
        resnet_feature_np = np.array(resnet_feature)
        resnet_feature_list.append(resnet_feature_np.flatten())
    return np.array(resnet_feature_list)
path = "/home/ubuntu/photos/*.jpeg"
array = extract_vector(path)
for i in range(3, 20):
    kmeans = KMeans(n_clusters=i, random_state=0).fit(array)
    lab = kmeans.labels_
    print(i, metrics.silhouette_score(array, kmeans.labels_, metric='euclidean'))

counter = 0
for x,y in zip(glob.glob(path), lab):
    counter += 1
    print(x,y)
    cv2.imwrite("/home/ubuntu/organized_photos/%s/%s.jpg" % (y, counter), cv2.imread(x))

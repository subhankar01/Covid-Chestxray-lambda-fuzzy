import os
from os.path import basename, join, exists
import numpy as np
np.random.seed(777)
import time
import keras as keras
import tensorflow as tf
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import decode_predictions
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,LSTM
from keras.layers import merge,Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os
from keras.preprocessing.image import ImageDataGenerator
# DCNN Models used
#---------------------------------------------
from keras.applications.inception_v3 import InceptionV3
#----------------------------------------------
from keras.applications.vgg16 import preprocess_input as pi_vgg16
from keras.applications.vgg19 import preprocess_input as pi_vgg19
from keras.applications.xception import preprocess_input as pi_xcep 
from keras.applications.inception_resnet_v2 import preprocess_input as pi_incepresnet
from keras.models import load_model
from numpy import array
from numpy import argmax
from sklearn.metrics import accuracy_score
from  numpy import mean 
from numpy import std
import matplotlib.pyplot as plt
from keras.optimizers import Adam,SGD
from keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint
from keras.layers import GlobalAveragePooling2D, Concatenate
from keras.layers import BatchNormalization,Dropout
from keras.layers import Lambda
from keras.regularizers import l2
import math
from keras import backend as K
from keras.metrics import categorical_accuracy
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
from keras.models import load_model

os.chdir(r"COVID_Xray/")

train_dir="aug/"
test_dir="test/"

total=0
print('---Training set details----')
for sub_folder in os.listdir(train_dir):
  no_of_images=len(os.listdir(train_dir + sub_folder))
  total+=no_of_images
  print(str(no_of_images) + " " + sub_folder + " images")

print("Total no. of Chest Xray training images=",total)

total=0
print('---Test set details----')
for sub_folder in os.listdir(test_dir):
  no_of_images=len(os.listdir(test_dir + sub_folder))
  total+=no_of_images
  print(str(no_of_images) + " " + sub_folder + " images")

print("Total no. of Chest Xray test images=",total)

extracted_features_dir="COVID_Xray/extracted_features/"

img_height =512
img_width =512
batch_size =32
input_shape = (img_width, img_height, 3)

print("-----------------Image Augmentation for InceptionV3--------------")

random_seed = np.random.seed(1142)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    validation_split= 0.2,
    zoom_range=0.1,
    shear_range=0.2)

train_generator_incep = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    seed = random_seed,
    shuffle=False,
    subset = 'training',
    class_mode='categorical')

val_generator_incep = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    seed = random_seed,
    shuffle=False,
    subset = 'validation',
    class_mode='categorical')

test_datagen=ImageDataGenerator(rescale=1./255)
test_generator_incep=test_datagen.flow_from_directory(test_dir,
                                                      target_size=(img_height, img_width),
                                                          batch_size=batch_size, 
                                                          seed=random_seed,
                                                          shuffle=False,
                                                          class_mode='categorical') 


nb_train_samples = len(train_generator_incep.filenames)
nb_validation_samples = len(val_generator_incep.filenames)
predict_size_train = int(math.ceil(nb_train_samples / batch_size))
predict_size_validation = int(math.ceil(nb_validation_samples / batch_size))

nb_test_samples = len(test_generator_incep.filenames)
predict_size_test = int(math.ceil(nb_test_samples / batch_size))

model_name="InceptionV3"
model = InceptionV3(include_top=False, weights="imagenet",pooling='avg',input_shape=input_shape)
image_input =model.input
x1 = GlobalAveragePooling2D()(model.get_layer("conv2d_3").output) #layer_11
x2 = GlobalAveragePooling2D()(model.get_layer("conv2d_8").output) #layer_18
x3 = GlobalAveragePooling2D()(model.get_layer("conv2d_5").output)  #layer_28
x4= GlobalAveragePooling2D()(model.get_layer("conv2d_12").output) #layer_51
x5= GlobalAveragePooling2D()(model.get_layer("conv2d_19").output) #layer_74
x6= GlobalAveragePooling2D()(model.get_layer("conv2d_34").output) #layer_101
x7= GlobalAveragePooling2D()(model.get_layer("conv2d_30").output) #layer_120
x8= GlobalAveragePooling2D()(model.get_layer("conv2d_40").output) #layer_152
x9= GlobalAveragePooling2D()(model.get_layer("conv2d_50").output) #layer_184
x10= GlobalAveragePooling2D()(model.get_layer("conv2d_60").output)#layer_216
x11= GlobalAveragePooling2D()(model.get_layer("conv2d_80").output) #layer_249
x12= GlobalAveragePooling2D()(model.get_layer("conv2d_76").output) #layer_263
x13= GlobalAveragePooling2D()(model.get_layer("conv2d_85").output)#layer_294
out= Concatenate()([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13])
custom_incep_model = Model(image_input , out)
custom_incep_model.summary()

for layer in custom_incep_model.layers[:249]:
    layer.trainable = False
custom_incep_model.summary()

#Saving features of the training images
features_train = custom_incep_model.predict_generator(train_generator_incep, predict_size_train)
np.save(extracted_features_dir+model_name+'_train_features.npy', features_train)

# Saving features of the validation images
features_validation = custom_incep_model.predict_generator(val_generator_incep, predict_size_validation)
np.save(extracted_features_dir+model_name+'_val_features.npy', features_validation)

# Saving features of the test images
features_test = custom_incep_model.predict_generator(test_generator_incep, predict_size_test)
np.save(extracted_features_dir+model_name+'_test_features.npy', features_test)


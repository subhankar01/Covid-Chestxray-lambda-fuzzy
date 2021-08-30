import os
from os.path import basename, join, exists
import numpy as np
import math
np.random.seed(777)
import tensorflow_addons as tfa
from tensorflow.keras import models
from tensorflow.keras import layers 
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras import callbacks
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from numpy import array
from numpy import argmax
from  numpy import mean 
from numpy import std
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')


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
img_width = 512
batch_size =32
input_shape = (img_width, img_height, 3)

print("-----------------Image Augmentation for Xception--------------")

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

train_generator_xcep = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    seed = random_seed,
    shuffle=False,
    subset = 'training',
    class_mode='categorical')

val_generator_xcep = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    seed = random_seed,
    shuffle=False,
    subset = 'validation',
    class_mode='categorical')

test_datagen=ImageDataGenerator(rescale=1./255)
test_generator_xcep=test_datagen.flow_from_directory(test_dir,
                                                      target_size=(img_height, img_width),
                                                          batch_size=batch_size, 
                                                          seed=random_seed,
                                                          shuffle=False,
                                                          class_mode='categorical') 

nb_train_samples = len(train_generator_xcep.filenames)
nb_validation_samples = len(val_generator_xcep.filenames)
predict_size_train = int(math.ceil(nb_train_samples / batch_size))
predict_size_validation = int(math.ceil(nb_validation_samples / batch_size))
nb_test_samples = len(test_generator_xcep.filenames)
predict_size_test = int(math.ceil(nb_test_samples / batch_size))

model_name="Xception"
model = Xception(include_top=False, weights="imagenet",pooling='avg',input_shape=input_shape)
image_input =model.input
x1 = layers.GlobalAveragePooling2D()(model.get_layer("block4_sepconv1").output) #layer_27
x2 = layers.GlobalAveragePooling2D()(model.get_layer("block5_sepconv1").output) #layer 37
x3 = layers.GlobalAveragePooling2D()(model.get_layer("block14_sepconv1").output)  #layer_126
out= layers.Concatenate()([x1,x2,x3])
out=layers.Dense(512,activation='relu')(out)
out=layers.Dropout(0.5)(out)
out=layers.Dense(3,activation='softmax',name= 'output')(out)
custom_xcep_model = models.Model(image_input , out)
custom_xcep_model.summary()

for layer in custom_xcep_model.layers[:115]:
    layer.trainable = False
custom_xcep_model.summary()

nEpochs=100
base_lr=1e-3

opt = optimizers.Adam(lr=base_lr, beta_1=0.6, beta_2=0.8,amsgrad=True)
custom_xcep_model.compile(optimizer = opt, loss='categorical_crossentropy', metrics=['accuracy'])
checkpoint1 = callbacks.ModelCheckpoint('saved models/Xception/xception_weights.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list=[checkpoint1]
# Training the modified Xception network for refining the deep feature embedding
history =custom_xcep_model.fit(train_generator_xcep,
                    epochs=nEpochs,
                    validation_data=val_generator_xcep,
                    callbacks=callbacks_list)
bottleneck= tf.keras.Model(inputs=custom_xcep_model.input, outputs=custom_xcep_model.layers[130].output)
#Saving features of the training images
features_train =bottleneck.predict_generator(train_generator_xcep, predict_size_train)
np.save(extracted_features_dir+model_name+'_train_features.npy', features_train)

# Saving features of the validation images
features_validation =bottleneck.predict_generator(val_generator_xcep, predict_size_validation)
np.save(extracted_features_dir+model_name+'_val_features.npy', features_validation)

# Saving features of the test images
features_test = bottleneck.predict_generator(test_generator_xcep, predict_size_test)
np.save(extracted_features_dir+model_name+'_test_features.npy', features_test)

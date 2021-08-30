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
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
x1 = layers.GlobalAveragePooling2D()(model.get_layer("conv2d_3").output) #layer_11
x2 = layers.GlobalAveragePooling2D()(model.get_layer("conv2d_8").output) #layer_18
x3 = layers.GlobalAveragePooling2D()(model.get_layer("conv2d_5").output)  #layer_28
x4= layers.GlobalAveragePooling2D()(model.get_layer("conv2d_12").output) #layer_51
x5= layers.GlobalAveragePooling2D()(model.get_layer("conv2d_19").output) #layer_74
x6= layers.GlobalAveragePooling2D()(model.get_layer("conv2d_34").output) #layer_101
x7= layers.GlobalAveragePooling2D()(model.get_layer("conv2d_30").output) #layer_120
x8= layers.GlobalAveragePooling2D()(model.get_layer("conv2d_40").output) #layer_152
x9= layers.GlobalAveragePooling2D()(model.get_layer("conv2d_50").output) #layer_184
x10= layers.GlobalAveragePooling2D()(model.get_layer("conv2d_60").output)#layer_216
x11= layers.GlobalAveragePooling2D()(model.get_layer("conv2d_80").output) #layer_249
x12= layers.GlobalAveragePooling2D()(model.get_layer("conv2d_76").output) #layer_263
x13= layers.GlobalAveragePooling2D()(model.get_layer("conv2d_85").output)#layer_294
out= layers.Concatenate()([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13])
out=layers.Dense(512,activation='relu')(out)
out=layers.Dropout(0.5)(out)
out=layers.Dense(3,activation='softmax',name= 'output')(out)
custom_incep_model = models.Model(image_input , out)
custom_incep_model.summary()

for layer in custom_incep_model.layers[:249]:
    layer.trainable = False
custom_incep_model.summary()

nEpochs=100
base_lr=1e-3
lr_min=0
alpha=1
def lr_scheduler(epoch):
    lr = math.fabs(lr_min + (1 + math.cos(1 * epoch * math.pi /nEpochs)) * (base_lr - lr_min) / 2.)
    print('lr: %f' % lr)
    return lr

contr_loss=tfa.losses.contrastive_loss

opt = optimizers.Adam(lr=base_lr, beta_1=0.6, beta_2=0.8,amsgrad=True)
custom_incep_model.compile(optimizer = opt, loss=['categorical_crossentropy',contr_loss], metrics=['accuracy'])
checkpoint1 = callbacks.ModelCheckpoint('saved models/InceptionV3/inception_weights.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
lr_decay = callbacks.LearningRateScheduler(schedule=lr_scheduler)
callbacks_list=[checkpoint1,lr_decay]
history =custom_incep_model.fit(train_generator_incep,
                    epochs=nEpochs,
                    validation_data=val_generator_incep,
                    callbacks=callbacks_list)
bottleneck= tf.keras.Model(inputs=custom_incep_model.input, outputs=custom_incep_model.layers[294].output)
#Saving features of the training images
features_train = bottleneck.predict_generator(train_generator_incep, predict_size_train)
np.save(extracted_features_dir+model_name+'_train_features.npy', features_train)

# Saving features of the validation images
features_validation = bottleneck.predict_generator(val_generator_incep, predict_size_validation)
np.save(extracted_features_dir+model_name+'_val_features.npy', features_validation)

# Saving features of the test images
features_test = vbottleneck.predict_generator(test_generator_incep, predict_size_test)
np.save(extracted_features_dir+model_name+'_test_features.npy', features_test)

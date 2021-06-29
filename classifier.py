import os
from os.path import basename, join, exists
import numpy as np
import math
np.random.seed(777)
import time
import tensorflow_addons as tfa
from tensorflow.keras import models
from tensorflow.keras import layers 
from tensorflow.keras import models
from tensorflow.keras import utils 
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
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

# Labels
train_labels=np.load('labels/train_labels.npy')
val_labels=np.load('labels/val_labels.npy')
test_labels=np.load('labels/test_labels.npy')

print(train_labels.shape)
print(val_labels.shape)
print(test_labels.shape)

# Feature embeddings
train_data=np.load('COVID_Xray/extracted_features/VGG16_train_features.npy')
val_data=np.load('COVID_Xray/extracted_features/VGG16_train_features.npy')
test_data=np.load('COVID_Xray/extracted_features/VGG16_train_features.npy')

print(train_data.shape)
print(val_data.shape)
print(test_data.shape)


nEpochs=1000
base_lr=1e-3
batch_size=64
lr_min=0
alpha=1
def lr_scheduler(epoch):
    lr = math.fabs(lr_min + (1 + math.cos(1 * epoch * math.pi /nEpochs)) * (base_lr - lr_min) / 2.)
    print('lr: %f' % lr)
    return lr

contr_loss=tfa.losses.contrastive_loss

# MLP architecture

model =models.Sequential()
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3,activation='softmax',name= 'output'))
opt = optimizers.Adam(lr=base_lr, beta_1=0.6, beta_2=0.8,amsgrad=True)
model.compile(optimizer = opt, loss=['categorical_crossentropy',contr_loss], metrics=['accuracy'])

checkpoint1 = callbacks.ModelCheckpoint('saved models/VGG16/mlp_weights.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
lr_decay = callbacks.LearningRateScheduler(schedule=lr_scheduler)
callbacks_list=[checkpoint1,lr_decay]

# MLP Training 
history = model.fit(train_data, train_labels,
                    epochs=nEpochs,
                    batch_size=batch_size,
                    validation_data=(val_data, val_labels),
                    callbacks=callbacks_list,
                    verbose= 2)

# Accuracy Curve
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc='lower right')
plt.show()

# Loss curve
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc='upper left')
plt.show()


model.save('saved models/VGG16/mlp.h5')
loaded_model=models.load_model('saved models/VGG16/mlp.h5',compile=False)
loaded_model.load_weights('saved models/VGG16/mlp_weights.h5')

#Validation acc
preds = loaded_model.predict(val_data)
predictions = [i.argmax() for i in preds]
y_true = [i.argmax() for i in val_labels]
print('Validation Accuracy={}'.format(accuracy_score(y_true=y_true, y_pred=predictions)))

# Test acc
preds = loaded_model.predict(val_data)
predictions = [i.argmax() for i in preds]
y_true = [i.argmax() for i in val_labels]
print('Validation Accuracy={}'.format(accuracy_score(y_true=y_true, y_pred=predictions)))


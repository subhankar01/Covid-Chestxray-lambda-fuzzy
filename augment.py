# image augmentation 
import os
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    shear_range=0.2,
    fill_mode='nearest',
    horizontal_flip=True)

folder=['Covid','Pneumonia','Normal']
for f in folder:
  parent_dir="train/"+f+"/"
  save_dir="aug/"+f+"/"
  print(len(os.listdir(dir)))
  for filename in os.listdir(parent_dir):
    file=parent_dir+filename
    fname=filename.split('.')[0]
    img = load_img(file)
    x = img_to_array(img) 
    x = x.reshape((1,) + x.shape) 
    i = 0
    for batch in datagen.flow(x, batch_size=1,save_to_dir=save_dir, save_prefix=fname, save_format='png'):
        if i==5:
            break
        i += 1
  

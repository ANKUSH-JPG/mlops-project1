#!/usr/bin/env python

from keras.models import Sequential
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Convolution2D
from keras.layers import Dense
from keras.optimizers import Adam

import yaml

model=Sequential()
model.add(Convolution2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(64,64,3)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.summary()
model.add(Dense(units=1024,activation='relu'))
model.add(Dense(units=512,activation='relu'))
model.add(Dense(units=250,activation='relu'))
model.add(Dense(units=75,activation='relu'))
model.add(Dense(units=25,activation='relu'))
model.add(Dense(units=10,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))

model.summary()
model.compile(optimizer=Adam(learning_rate=0.01),loss='binary_crossentropy',metrics=['accuracy'])

from keras_preprocessing.image import ImageDataGenerator

config_vals = ""
with open("config.yaml", "r") as cr:
   config_vals = yaml.load(cr)

spc = config_vals['spc']
ep = config_vals['ep']
print("steps per epoch = "+ str(spc))
print("epochs = "+ str(ep))

config_vals['spc'] = spc + 100

with open("config.yaml", "w") as cw:
   yaml.dump(config_vals, cw, default_flow_style=True)


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_dataset = train_datagen.flow_from_directory(
        '/covid_folder/xray_dataset_covid19/train/',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')
test_dataset = test_datagen.flow_from_directory(
        '/covid_folder/xray_dataset_covid19/test/',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')
working=model.fit(
        train_dataset,
        steps_per_epoch=spc,
        epochs=ep,
        validation_data=test_dataset,
        validation_steps=800)


model.save('covid19_predictor.h5')

train_dataset.class_indices

accuracy_of_model=str(working.history.get('accuracy')[-1])
type(accuracy_of_model)

file=open('accuracy.txt','w')
file.write(accuracy_of_model)
file.close()

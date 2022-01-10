# -*- coding: utf-8 -*-
###############################################################################
# Mission Description : ResNet-50 for CIFAR10 Classification
# Author of Codes     : Yuzhe Li (李煜喆)
# Date of Completion  : 2022-01-09
# Contact of Author   : 2233560443@qq.com (QQ: 2233560443)
# Technical Backups   : Jingyi Yang (杨景懿); Yuqun Lin (林育群); Guangsen Zhang (张广森)
###############################################################################
# Materials and Tools to Build Individual CNN Networks
import tensorflow as tf
from tensorflow.keras import Model, regularizers
from tensorflow.keras.datasets  import cifar10
from tensorflow.keras.layers    import Conv2D, MaxPool2D, AveragePooling2D, GlobalAveragePooling2D, Lambda
from tensorflow.keras.layers    import Input, BatchNormalization, Activation, Dropout, Dense, concatenate
from tensorflow.keras.utils     import plot_model
from sklearn.model_selection    import train_test_split
# API of Image Enhancement and image data converters
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
# Tools for Visualization
import os
import matplotlib.pyplot as plt
import numpy
import shutil
###############################################################################
real_time_epochs   = 15

img_rows, img_cols = 32, 32
img_channels       = 3
weight_decay       = 1e-4

mother_path = './Storage_Resnet-50/'
Order       = True
###############################################################################
# Preparation and Preprocessing of Datasets (CIFAR10: 10-Classification RGB Figures)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train.shape) # -> (50000, 32, 32, 3)
# print(y_train.shape) # -> (50000, 1)
# print(x_test.shape)  # -> (50000, 32, 32, 3)
# print(y_test.shape)  # -> (10000, 1)
seed_value = 108
numpy.random.seed(seed_value)
numpy.random.shuffle(x_train)
numpy.random.seed(seed_value)
numpy.random.shuffle(y_train)
# tf.random.set_seed(seed_value)
labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
###############################################################################
# Function to Display a Proportion of Pictures Included in the Original Training Set
def imageDisplay(x_dataset, y_dataset, canvas_height, canvas_length):
    plt.figure(figsize=(10, 8))
    plt.suptitle('Image Samples of CIFAR10')
    for i in range(canvas_height*canvas_length):
        plt.subplot(canvas_height, canvas_length, (i+1))
        plt.imshow(x_dataset[i])
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(labels[y_dataset[i][0]])
    plt.show()
    # imageDisplay(x_train, y_train, 4, 6)

# Function to Generate and Enhance Image Data
def imageGenerator(x_dataset, y_dataset, flag, seed_value):
    if flag == False:
        return x_dataset, y_dataset
    if flag == True:
        original_train_figure_number = x_dataset.shape[0]
        # print(original_train_figure_number)
        # Image Data Reinforcement
        x_dataset = numpy.concatenate((x_dataset, x_dataset), axis=0)
        y_dataset = numpy.concatenate((y_dataset, y_dataset), axis=0)
        for i in range(original_train_figure_number):
            x_dataset[i] = tf.image.random_flip_left_right(x_dataset[i])
            x_dataset[i] = tf.image.random_flip_up_down   (x_dataset[i])
            x_dataset[i] = tf.image.random_contrast  (x_dataset[i], 0.8, 1.5)
            x_dataset[i] = tf.image.random_hue       (x_dataset[i], 0.1)
            x_dataset[i] = tf.image.random_saturation(x_dataset[i], 0.8, 1.5)
            if i % 1000 == 0:
                print(i)
        numpy.random.seed(seed_value)
        numpy.random.shuffle(x_dataset)
        numpy.random.seed(seed_value)
        numpy.random.shuffle(y_dataset)
        # tf.random.set_seed(seed_value)
        return x_dataset, y_dataset
    
# Normalization Function for x Datasets
def dataNormalize(x):
    x = x.astype('float32')
    x_normalize = x / 255.0
    return x_normalize

# Split the Original Training Set into a New Training Set and a Validation Set, then Normalize 
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.20)
# Image Data Generation and Enhancement
x_train, y_train = imageGenerator(x_train, y_train, flag=False, seed_value=150)
# Image Data Normalization
x_train = dataNormalize(x_train)
x_valid = dataNormalize(x_valid)
x_test  = dataNormalize(x_test )
# print(x_train.shape)
# print(y_train.shape)
# print(x_valid.shape)
# print(y_valid.shape)
cat_number = int(numpy.sum(numpy.max(y_train, 0))) + 1
# Convert the Label Value (y) of All Samples to One-Hot Vectors
y_train = tf.keras.utils.to_categorical(y_train, cat_number)
y_valid = tf.keras.utils.to_categorical(y_valid, cat_number)
y_test  = tf.keras.utils.to_categorical(y_test , cat_number)
##############################################################################
# Resnet Method
def conv(x, output_filters, kernel_size, strides):
    return Conv2D(filters=output_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(weight_decay),
                  use_bias=False)(x)

def DenseLayer(x, neuron_number, activation_func):
    return Dense(units=neuron_number,
                 activation=activation_func,
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(x)

# Functions to Define the Topology Structure of Bottleneck Blocks Layers in ResNeXt
def BottleNeckBlock(x, nchannels, strides, residual_path):
    residual  = x
    x = conv(x, output_filters=1*nchannels, kernel_size=1, strides=strides)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = conv(x, output_filters=1*nchannels, kernel_size=3, strides=1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = conv(x, output_filters=2*nchannels, kernel_size=1, strides=1)
    x = BatchNormalization()(x)
    if residual_path == True:
        residual = conv(residual, output_filters=2*nchannels, kernel_size=1, strides=strides)
        residual = BatchNormalization()(residual)
        x = Activation('relu')(x + residual)
    else:
        x = Activation('relu')(x)
    outchannels = 2*nchannels
    return x, outchannels

# Functions to Define the Topology Structure of Bottleneck Blocks Layers in ResNeXt
def ResNetBlock(x, nchannels, strides, residual_path, block_dense):
    x, outchannels = BottleNeckBlock(x, nchannels, strides=strides, residual_path=True)
    for i in range(1, block_dense):
        x, outchannels = BottleNeckBlock(x, nchannels, strides=1, residual_path=True)
    return x, outchannels

# Define the Topology Structure of a Complete Resnet (Resnet18)
def ResNet(img_input, cat_num, block_list):
    # img_nchannels = 3
    nchannels   = 64
    numBlock    = len(block_list)
    x = conv(img_input, output_filters=nchannels, kernel_size=3, strides=1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    outchannels = nchannels
      
    for indBlock in range(0, 1):
        x, outchannels = ResNetBlock(x, outchannels, strides=1, residual_path=True, block_dense=block_list[indBlock]) 
    for indBlock in range(1, numBlock):
        x, outchannels = ResNetBlock(x, outchannels, strides=2, residual_path=True, block_dense=block_list[indBlock])
    
    hidden_neuron_number = 640
    hidden_layers_number = 2
    x = GlobalAveragePooling2D()(x)
    for i in range(hidden_layers_number):
        x = DenseLayer(x, hidden_neuron_number, 'elu')
    x = DenseLayer(x, 10, 'softmax')
    return x
###############################################################################
# File Operator Serving for the Model Loader
if os.path.exists(mother_path):
    if Order == True:
        shutil.rmtree(mother_path)
        os.mkdir(mother_path)
    else:
        pass
else:
    os.mkdir(mother_path)
# Function to Define a Dynamic Learning Rate for Training Process
def scheduler(epoch):
    if os.path.exists(mother_path + 'learning_rate.txt'):
        file = open(mother_path + 'learning_rate.txt', 'r+')
        initial_learning_rate = float(file.read())
        # print(initial_learning_rate)
        file.close()
    else:
        initial_learning_rate = 0.001
    decay_rate=0.99
    if epoch < 1 :
        learning_rate =  initial_learning_rate
        tf.keras.backend.set_value(model.optimizer.lr, learning_rate)
        file = open(mother_path + 'learning_rate.txt', 'w')
        file.write(str(learning_rate))
        file.close()
    else:
        learning_rate =  initial_learning_rate * decay_rate
        tf.keras.backend.set_value(model.optimizer.lr, learning_rate)
        file = open(mother_path + 'learning_rate.txt', 'w')
        file.write(str(learning_rate))
        file.close()
    print('##################################################################')
    print("The Current Learning Rate: {}".
          format(tf.keras.backend.get_value(model.optimizer.lr)))
    print('##################################################################')
    return tf.keras.backend.get_value(model.optimizer.lr)
reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
###############################################################################
# Initialize the Topology Connection of Densenet Convolution Networks
tf.config.experimental_run_functions_eagerly(True)

img_input = Input(shape=(img_rows, img_cols, img_channels))
output    = ResNet(img_input, cat_number, [3, 4, 6, 3])
model     = Model(img_input, output)
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',metrics=['categorical_accuracy'])
model.summary()

# Check Point and Continue Training (Directly Load the Optinum Model Already Saved)
checkpoint_save_path = mother_path + 'checkpoint/Baseline.ckpt'
if os.path.exists(checkpoint_save_path + '.index'):
    print('----------------------Previous Model Parameters have been Reloaded----------------------')
    model.load_weights(checkpoint_save_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)
###############################################################################
# Training of Built Networks 
datagen = ImageDataGenerator(horizontal_flip=True,
                             width_shift_range=0.125,
                             height_shift_range=0.125,
                             fill_mode='constant',cval=0.)
datagen.fit(x_train)
# Training Process of Model (The Main Calculation and Time Consumer Part in Paragram )
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    epochs=real_time_epochs, validation_data=(x_valid, y_valid),
                    validation_freq=1, callbacks=[cp_callback, reduce_lr])

# Evaluation and Test of Model on Test Datasets that have been Prepared in CIFAR10
eval_result = model.evaluate(x_test, y_test)
# Store Parameters of the Best Trained Model
file = open(mother_path + 'weights.txt','w')
for content in model.trainable_variables :
    file.write(str(content.name   ) + '\n')
    file.write(str(content.shape  ) + '\n')
    file.write(str(content.numpy()) + '\n')
file.close()
###############################################################################
train_acc  = history.history['categorical_accuracy'    ]
valid_acc  = history.history['val_categorical_accuracy']
train_loss = history.history['loss'    ]
valid_loss = history.history['val_loss']
file = open(mother_path + 'train_acc.txt','a+')
for content in train_acc :
    file.write(str(content) + '\n')
file.close()
file = open(mother_path + 'valid_acc.txt','a+')
for content in valid_acc :
    file.write(str(content) + '\n')
file.close()
file = open(mother_path + 'train_loss.txt','a+')
for content in train_loss :
    file.write(str(content) + '\n')
file.close()
file = open(mother_path + 'valid_loss.txt','a+')
for content in valid_loss:
    file.write(str(content) + '\n')
file.close()
# plt.subplot(1, 2, 1)
# plt.plot(train_acc, label='Training Accuracy'  )
# plt.plot(valid_acc, label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()
# plt.subplot(1, 2, 2)
# plt.plot(train_loss, label='Training Loss'  )
# plt.plot(valid_loss, label='Validation Loss')
# plt.title('Training and Validation Test Loss')
# plt.legend()
# plt.show()                                                                                                                                                                                                                                                                                                 
print('Hello')
print('Loss     on Test Dataset:',eval_result[0]);
print('Accuracy on Test Dataset:',eval_result[1]);
file = open(mother_path + 'test_loss.txt','w')
file.write(str(eval_result[0]) + '\n')
file.close()
file = open(mother_path + 'test_acc.txt','w')
file.write(str(eval_result[1]) + '\n')
file.close()    
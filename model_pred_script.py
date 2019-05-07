import keras
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D,Dropout,Flatten
from keras.applications.mobilenet import preprocess_input
import numpy as np
import csv
import os
from keras.utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from keras.callbacks import History
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

#Load saved models
model1 = load_model("nasnet.h5", custom_objects={'auc': auc})
model2 = load_model("nasnet_randTEST.h5", custom_objects={'auc': auc}, compile=False)
#('auc': auc})
#####SESSION VARIABLES!!!######
all_batch_size = 40
#num_of_epochs = 10
#adam_op = Adam(lr=0.001)
test_file_path = '/home/victor/venv1/keras/data/test'
image_size = 224
test_file_stats = "nasnetTEST_prediction_stats.txt"
rescale_img=1./255
##############################

clear_file = open(test_file_stats, "w+")
clear_file.close()

####TEST DATA SETUP####
"""
model = load_model(model_name, custom_objects={'auc': auc})
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc', auc])	
"""

# Compile both pre-trained and random models
"""
model1.compile(loss='binary_crossentropy',
              optimizer=adam_op,
              metrics=['acc'])

model2.compile(loss='binary_crossentropy',
              optimizer=adam_op,
              metrics=['acc'])
"""
#Setup data generator and test_generator
test_datagen=ImageDataGenerator(rescale=rescale_img)

test_generator=test_datagen.flow_from_directory(test_file_path,
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=all_batch_size,
                                                 class_mode='categorical',
												 shuffle = False)

#Define test_generator step size
step_size_test=test_generator.samples/test_generator.batch_size

#Predict both models against test set
model1_Y_pred = model1.predict_generator(test_generator, step_size_test, verbose=1)
model2_Y_pred = model2.predict_generator(test_generator, step_size_test, verbose=1)

model1_y_pred = np.argmax(model1_Y_pred, axis=1)
model2_y_pred = np.argmax(model2_Y_pred, axis=1)

"""
#OPTIONAL PRINT STATEMENTS!!!

print("\ndensenet_y_pred:")
print(densenet_y_pred)
print('Classes DenseNet (correct)')
print(test_generator.classes)
print("\n")

print("\ndensenet_rand_y_pred:")
print(densenet_rand_y_pred)
print('Classes DenseNet_rand (correct)')
print(test_generator.classes)

####
"""

#Rename predictions for clarity
model1_pred = model1_y_pred
model2_pred = model2_y_pred
correct_pred = test_generator.classes

#Python array instansiation is AMAZING!?
output = [None] * 4
for i in range(len(output)):
    output[i] = 0

#populate output[]
for i in range(len(correct_pred)):
    if model1_pred[i] == model2_pred[i] and model1_pred[i] == correct_pred[i]:
        output[0] += 1
    if model1_pred[i] != correct_pred[i] and model2_pred[i] == correct_pred[i]:
        output[1] += 1
    if model2_pred[i] != correct_pred[i] and model1_pred[i] == correct_pred[i]:
        output[2] += 1
    if model1_pred[i] == model2_pred[i] and model1_pred[i] != correct_pred[i]:
        output[3] += 1

#Open and write output[] comma separated to file
f = open(test_file_stats, "w+")
f.write("Columns 1. Both models = correct, 2. model 1 (rand model)= wrong, model 2 (pre-trained model)= wrong, 4. both models = wrong\n")

output_str = ""
for i in range(len(output)):
    output_str += str(output[i]) + ","

output_str = output_str[:-1]
f.write(output_str)

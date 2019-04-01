"""
Densenet modell testkörd för både random och imagenetvärden
Skapa testgenerator
Batchsize måste vara delbar med antal bilder till generator
Sätt train_batchsize, test_batchsize, val_batchsize
"""
from __future__ import print_function
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.applications.nasnet import NASNetLarge
from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.metrics import auc as sk_auc
import csv
import os
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from keras import backend as K

train_dir = 'data/train'
validation_dir = 'data/validation'
test_dir = 'data/test'

image_size = 331
model_name = "all_freezed_nasnetlarge_smalltest_softmax.h5"
#Load the VGG model
#SET weights
nasnet_conv = NASNetLarge(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

# Freeze all the layers
# :-n, n = # of open layers
for layer in nasnet_conv.layers[:]:
    #SET trainable
	#Ev trainable lager för transfer
	layer.trainable = False

# Check the trainable status of the individual layers
for layer in nasnet_conv.layers:
    print(layer, layer.trainable)

# Create the model
model = models.Sequential()

# Add the vgg convolutional base model
model.add(nasnet_conv)

# Add new layers
model.add(layers.Flatten())
#kernel_regularizer=regularizers.l2(0.001)
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='sigmoid'))

# model.summary()
##2##
# No Data augmentation 
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

# Change the batchsize according to your system RAM
train_batchsize = 100
val_batchsize = 100

# Data Generator for Training data
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=train_batchsize,
        class_mode='categorical',
		shuffle=True)

# Data Generator for Validation data
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=True)

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc', auc])

# Train the Model

history = model.fit_generator(
      train_generator,
      steps_per_epoch=train_generator.samples/train_generator.batch_size ,
      epochs=4,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
      verbose=1)

# Save the Model
model.save(model_name)
# Plot the accuracy and loss curves
acc = history.history['acc']
val_acc = history.history['val_acc']
auc_hist = history.history['auc']
val_auc = history.history['val_auc']
loss = history.history['loss']
val_loss = history.history['val_loss']
#print(acc)
#print(val_acc)
epochs = range(len(acc))

#Write result metrics to file	
def writetrainmetrics(met):
	#filename blir resultmetrics_ + namnet på .py filen koden finns i + txt
	filename = "trainmetrics_" + os.path.basename(__file__).split(".")[0] + ".txt"
	with open(filename,"a", newline="\r\n") as f:
		writer = csv.writer(f, lineterminator=os.linesep)
		if os.path.getsize(filename) == 0:
			f.write("Rows: 1. Acc, 2. val_acc, 3. loss, 4. val_loss, 5. auc, 6. val_auc" + "\n")	
		curr = 0
		lenacc = len(met)
		writer.writerow(met)
		f.close()

def writetestmetrics(met, label):
	#filename blir resultmetrics_ + namnet på .py filen koden finns i + txt
	filename = "testmetrics_" + os.path.basename(__file__).split(".")[0] + ".txt"
	with open(filename,"a", newline="\r\n") as f:
		writer = csv.writer(f, lineterminator=os.linesep)
		if os.path.getsize(filename) != 0:
			f.write("\r\n")
		else:
			f.write("Header: Testmetrics" + "\r\n")
		f.write(label + "\r\n")	
		f.write(met)	
		f.close()		

print("Writing train metrics...")		
writetrainmetrics(acc)
writetrainmetrics(val_acc)
writetrainmetrics(loss)
writetrainmetrics(val_loss)
writetrainmetrics(auc_hist)
writetrainmetrics(val_auc)

########################################
#####Implementation of fullmodel.py#####
########################################


#Load the VGG model from disk
#SET weights
model = load_model(model_name, custom_objects={'auc': auc})
test_datagen = ImageDataGenerator(rescale=1./255)

# Change the batchsize according to your system RAM
test_batchsize = 20
	
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(image_size, image_size),
        batch_size=test_batchsize,
        class_mode='categorical',
        shuffle=False)		

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc', auc])		
				
###################
###Skriv till fil##
###################

#generator_steps = antal ad + nonad images / batchsize i validate eller test
#32
generator_steps = 48
#Skapa test_generator
print("Starting test...")
Y_pred = model.predict_generator(test_generator, generator_steps, verbose=1) 
y_pred = np.argmax(Y_pred, axis=1)

##Ta endast positiva värden till AUC##
# Test shuffle av array om ej funkar
y_pred_pos = Y_pred[:, 1]
print("y_pred")
print(y_pred)
print("y_pred_pos")
print(y_pred_pos)
print('Classes')
print(test_generator.classes)
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))
tn, fp, fn, tp = confusion_matrix(test_generator.classes, y_pred).ravel()
acc_val_test = (tn+tp)/(tn+fp+fn+tp)
print('Classification Report')
target_names = ['0 - nonad', '1 - ad']
print(classification_report(test_generator.classes, y_pred, target_names=target_names))

#Till ROC#
fpr, tpr, threshold = roc_curve(test_generator.classes, y_pred_pos)
#AUC-VALUE

##UPPDATERA##
auc_val = roc_auc_score(test_generator.classes, y_pred)
#sk_auc_val = sk_auc(fpr, tpr)
#print("AUC: ")
#print(auc_val)
#print(sk_auc(fpr, tpr))
#auc1, update_op = tf.metrics.auc(test_generator.classes, y_pred)

"""
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    print("tf auc: {}".format(sess.run([auc1, update_op])))
"""	
print(fpr)
print(tpr)

#print(nyauc)
print("Accuracy: ")
print(acc_val_test)


#Testa
writetestmetrics(str(confusion_matrix(test_generator.classes, y_pred)), "Confusion Matrix")
writetestmetrics(str(classification_report(test_generator.classes, y_pred, target_names=target_names)), "Classification Report")
writetestmetrics(str(auc_val), "AUC: ")
writetestmetrics(str(acc_val_test), "Accuracy: ")
writetestmetrics(str(fpr), "FPR: ")
writetestmetrics(str(tpr), "TPR: ")
writetestmetrics(str(threshold), "Threshold: ")

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--', label='AUC (area = {:.3f})'.format(auc_val))
plt.plot(fpr, tpr)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
"""
plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc_val))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')
plt.show()

"""





"""
#tn, fp, fn, tp = confusion_matrix(validation_generator.classes, y_pred).ravel()
#print(tn, fp, fn, tp)
	
#Plot result metrics
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
"""


#fpr_keras, tpr_keras, thresholds_keras = roc_curve(validation_generator, y_pred_keras, pos_label="ad")
#auc_keras = auc(fpr_keras, tpr_keras)


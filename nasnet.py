import keras
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D,Dropout,Flatten
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.applications.nasnet import NASNetMobile
from keras.applications.mobilenet import preprocess_input
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from keras.utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from keras.callbacks import History
import tensorflow as tf
"""
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
"""
#####SESSION VARIABLES!!!#####
f = open("trainmetrics_nasnet.txt", "w+")
all_batch_size = 40
num_of_epochs = 10
adam_op = keras.optimizers.Adam(lr=0.0001)
train_file_path = 'data/train'
validation_file_path = 'data/validation'
test_file_path = 'data/test'
image_size = 224
model_name = "nasnet.h5"
test_file_stats = "testmetrics_nasnet.txt"
##############################

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

#DECLARE MODEL weights='imagenet', include_top=false...
base_model=NASNetMobile(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3)) #imports the resnet model and discards the last 1000 neuron layer.

x=base_model.output
#x=GlobalAveragePooling2D()(x)
x=Flatten()(x)
#x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dropout(0.5)(x)
#x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(2,activation='sigmoid')(x) #final layer with sigmoid activation

#specify the inputs
#specify the outputs
model=Model(inputs=base_model.input,outputs=preds)
#now a model has been created based on our architecture

#TRAINABLE/NON-TRAINABLE LAYERS
for layer in model.layers[:769]:
    layer.trainable = False
 
"""
for layer in model.layers[:175]:
    layer.trainable=False
"""
for i,layer in enumerate(model.layers):
    print(i,layer.name,layer.trainable)
	
print()

#TRAINING SET ITERATOR
train_datagen=ImageDataGenerator(rescale=1./255) #, preprocessing_function=preprocess_input) #included in our dependencies

train_generator=train_datagen.flow_from_directory(train_file_path,
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=all_batch_size,
                                                 class_mode='categorical',
                                                 shuffle=True)




step_size_train=train_generator.samples/train_generator.batch_size


#VALIDATION SET ITERATOR
valid_datagen=ImageDataGenerator(rescale=1./255) #preprocessing_function=preprocess_input) #included in our dependencies

valid_generator=valid_datagen.flow_from_directory(validation_file_path,
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=all_batch_size,
                                                 class_mode='categorical',
												 shuffle=True)




step_size_valid=valid_generator.samples/valid_generator.batch_size #STEPS PER EPOCH //STEPS * ALL_BATCH_SIZE

model.compile(optimizer=adam_op,
                   loss='binary_crossentropy',
				   metrics=['acc', auc])

history = model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   epochs=num_of_epochs,
                   validation_data= valid_generator,
                   validation_steps=step_size_valid,
                   verbose = 1)


model.save(model_name)

#plot_model(model, to_file='model.png')

print(history.history.keys(), '\n')
# summarize history for accuracy

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
auc_hist = history.history['auc']
val_auc = history.history['val_auc']

f.write("1.acc, 2.val_acc, 3. loss, 4.val_loss, 5. auc, 6. val_auc\n")

acc_line = ''
val_acc_line = ''
loss_line = ''
val_loss_line = ''
auc_line = ''
val_auc_line = ''


i = 0
while i < len(val_acc):
    acc_line +=  str(acc[i]) + ','
    val_acc_line += str(val_acc[i]) + ','
    loss_line += str(loss[i]) + ','
    val_loss_line += str(val_loss[i]) + ','
    auc_line += str(auc_hist[i]) + ','
    val_auc_line += str(val_auc[i]) + ','
    i += 1

acc_line = acc_line[:-1]
val_acc_line = val_acc_line[:-1]
loss_line = loss_line[:-1]
val_loss_line = val_loss_line[:-1]
auc_line = auc_line[:-1]
val_auc_line = val_auc_line[:-1]

f.write(acc_line + '\n')
f.write(val_acc_line + '\n')
f.write(loss_line + '\n')
f.write(val_loss_line + '\n')
f.write(auc_line + '\n')
f.write(val_auc_line)


####TEST DATA SETUP####
def writetestmetrics(met, label):
	filename = test_file_stats
	with open(filename, "a") as f:
		if os.path.getsize(filename) != 0:
			f.write("\n")
		else:
			f.write("Header: Testmetrics" + "\n")
		f.write(label + "\n")
		f.write(met)
		f.close()

"""
model = load_model(model_name, custom_objects={'auc': auc})
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc', auc])
"""

test_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

test_generator=test_datagen.flow_from_directory(test_file_path,
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=all_batch_size,
                                                 class_mode='categorical',
												 shuffle = False)

step_size_test=test_generator.samples/test_generator.batch_size

Y_pred = model.predict_generator(test_generator, step_size_test, verbose=1)
y_pred = np.argmax(Y_pred, axis=1)

#Confusion Matrix and Classification Report
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
target_names = ['ad', 'nonad']
print(classification_report(test_generator.classes, y_pred, target_names=target_names))

f.close()

#Till ROC#
fpr, tpr, threshold = roc_curve(test_generator.classes, y_pred_pos)
#AUC-VALUE

##UPPDATERA##
auc_val = roc_auc_score(test_generator.classes, y_pred)

print("fpr", fpr)
print("tpr", tpr)

#print(nyauc)
print("Accuracy: ")
print(acc_val_test)

clear_file = open(test_file_stats, 'w+')
clear_file.close()

writetestmetrics(str(confusion_matrix(test_generator.classes, y_pred)), "Confusion Matrix")
writetestmetrics(str(classification_report(test_generator.classes, y_pred, target_names=target_names)), "Classification Report")
writetestmetrics(str(auc_val), "AUC: ")
writetestmetrics(str(acc_val_test), "Accuracy: ")
"""
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--', label='AUC (area = {:.3f})'.format(auc_val))
plt.plot(fpr, tpr)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
"""

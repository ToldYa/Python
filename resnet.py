import keras
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet import preprocess_input
import numpy as np
from IPython.display import Image
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import History


#####SESSION VARIABLES!!!#####
f = open("resnet_stats.txt", "w+")
all_batch_size = 20
num_of_epochs = 5
rms_op = keras.optimizers.RMSprop(lr=1e-3)
train_file_path = 'data/train'
validation_file_path = 'data/validation'
test_file_path = 'data/test'
all_layers_trainable = False
##############################

def specificity(y_pred, y_true):
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fp = K.sum(neg_y_true * y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    specificity = tn / (tn + fp + K.epsilon())
    return specificity

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
#DECLARE MODEL weights='imagenet', include_top=false...
base_model=ResNet50(weights='imagenet', include_top=False) #imports the resnet model and discards the last 1000 neuron layer.

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(2,activation='sigmoid')(x) #final layer with sigmoid activation

#specify the inputs
#specify the outputs
model=Model(inputs=base_model.input,outputs=preds)
#now a model has been created based on our architecture

#TRAINABLE/NON-TRAINABLE LAYERS
if not all_layers_trainable:
    for layer in model.layers[:175]:
        layer.trainable = False
 
"""
for layer in model.layers[:175]:
    layer.trainable=False
"""
for i,layer in enumerate(model.layers):
    print(i,layer.name,layer.trainable)
	
print()

#TRAINING SET ITERATOR
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

train_generator=train_datagen.flow_from_directory(train_file_path,
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=all_batch_size,
                                                 class_mode='categorical',
                                                 shuffle=True)




step_size_train=train_generator.n//train_generator.batch_size


#VALIDATION SET ITERATOR
valid_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

valid_generator=valid_datagen.flow_from_directory(validation_file_path,
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=all_batch_size,
                                                 class_mode='categorical')




step_size_valid=valid_generator.n//valid_generator.batch_size #STEPS PER EPOCH //STEPS * ALL_BATCH_SIZE

model.compile(optimizer=rms_op,loss='binary_crossentropy',metrics=['accuracy', specificity, recall])

history = model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   epochs=num_of_epochs,
				   validation_data= valid_generator,
				   validation_steps=step_size_valid)



#plot_model(model, to_file='model.png')

print(history.history.keys(), '\n')
# summarize history for accuracy
loss_hist = history.history['loss']
val_loss_hist = history.history['val_loss']
acc_hist = history.history['acc']
val_acc_hist = history.history['val_acc']
f.write("1.acc, 2.val_acc, 3. loss, 4.val_loss\n")

acc_line = ''
val_acc_line = ''
loss_line = ''
val_loss_line = ''


i = 0
while i < len(loss_hist):
    acc_line +=  str(acc_hist[i]) + ','
    val_acc_line += str(val_acc_hist[i]) + ','
    loss_line += str(loss_hist[i]) + ','
    val_loss_line += str(val_loss_hist[i]) + ','
    i += 1

val_loss_line = val_loss_line[:-1]
val_acc_line = val_acc_line[:-1]

f.write(val_acc_line + '\n')
f.write(val_loss_line)

####TEST DATA SETUP####

test_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

test_generator=test_datagen.flow_from_directory(test_file_path,
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=all_batch_size,
                                                 class_mode='categorical',
												 shuffle = False)

step_size_test=test_generator.n//test_generator.batch_size

#Confusion Matrix and Classification Report
Y_pred = model.predict_generator(test_generator, step_size_test, verbose=1)
y_pred = np.argmax(Y_pred, axis=1)
print('\n####################\nConfusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))

print('Classification Report')
target_names = ['ad', 'nonad']
print(classification_report(test_generator.classes, y_pred, target_names=target_names))

f.write('\n\nConfusion Matrix\n')
f.write(str(confusion_matrix(test_generator.classes, y_pred)))
f.write('\n\nClassification Report\n')
f.write(str(classification_report(test_generator.classes, y_pred, target_names=target_names)))

f.close()

import numpy as np
from tensorflow.keras.callbacks import  ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
import gc
from tensorflow.keras.preprocessing import image as IMG
from load_data import load_data
from build_model import build_model

# Data preprocessing
benign_train_dataset = np.array(load_data('datasets/training/benign',224))
malign_train_dataset = np.array(load_data('datasets/training/malignant',224))
benign_test_dataset = np.array(load_data('datasets/testing/benign',224))
malign_test_dataset = np.array(load_data('datasets/testing/malignant',224))
benign_train_dataset_label = np.zeros(len(benign_train_dataset))
malign_train_dataset_label = np.ones(len(malign_train_dataset))
benign_test_dataset_label = np.zeros(len(benign_test_dataset))
malign_test_dataset_label = np.ones(len(malign_test_dataset))
X_train_dataset = np.concatenate((benign_train_dataset, malign_train_dataset), axis = 0)
Y_train_dataset_label = np.concatenate((benign_train_dataset_label, malign_train_dataset_label), axis = 0)
X_test_dataset = np.concatenate((benign_test_dataset, malign_test_dataset), axis = 0)
Y_test_dataset_label = np.concatenate((benign_test_dataset_label, malign_test_dataset_label), axis = 0)
shuffle = np.arange(X_train_dataset.shape[0])
np.random.shuffle(shuffle)
X_train_dataset = X_train_dataset[shuffle]
Y_train_dataset_label = Y_train_dataset_label[shuffle]
shuffle = np.arange(X_test_dataset.shape[0])
np.random.shuffle(shuffle)
X_test_dataset = X_test_dataset[shuffle]
Y_test_dataset_label = Y_test_dataset_label[shuffle]
Y_train_dataset_label = to_categorical(Y_train_dataset_label, num_classes= 2)
Y_test_dataset_label = to_categorical(Y_test_dataset_label, num_classes= 2)
x_train_dataset, x_validation_dataset, y_train_dataset_label, y_validation_dataset_label = train_test_split(X_train_dataset,Y_train_dataset_label,test_size=0.2,random_state=11)
train_generator = ImageDataGenerator(rescale = 1.0/255,zoom_range=2,rotation_range = 90,horizontal_flip=True, vertical_flip=True)

# save checkpoint 
K.clear_session()
gc.collect()
learn_control = ReduceLROnPlateau(monitor='val_acc', patience=5,verbose=1,factor=0.2, min_lr=1e-7)
filepath="weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# model training
model = build_model()
batch_size = 18
validation_dataset = (x_validation_dataset, y_validation_dataset_label)
spe = x_train_dataset.shape[0] / batch_size
history = model.fit_generator(train_generator.flow(x_train_dataset, y_train_dataset_label, batch_size=batch_size),steps_per_epoch= spe,epochs=20,validation_data=validation_dataset,callbacks=[learn_control, checkpoint])

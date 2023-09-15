import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, utils
from sklearn.preprocessing import LabelEncoder
from PoseEMOT_Models import param_grid, create_transformer_model, create_vl_lstm_model

poses = np.load("PoseEncodedGEMEP_FULL.npy")
labels = np.load("PoseLabelsGEMEP_FULL.npy")
max_length = max_length = max([video.shape[0] for video in poses])
print(max_length)
# put data into 2 categories (positive and negative)
one_hot_labels = []
for label in labels:
    if label == "amu" or label == "joy" or label == "ple" or label == "rel" or label == "pri" or label == "ten" or label == "adm":
        one_hot_labels.append([1,0])
    else:
        one_hot_labels.append([0,1])

one_hot_labels = np.array(one_hot_labels)
# Split data into training and testing sets
count = labels.shape[0]
train_count = int(count * 0.8)
test_count = count - train_count
train_data = poses[:train_count]
train_labels = one_hot_labels[:train_count]

test_data = poses[train_count:]
test_labels = one_hot_labels[train_count:]

# Normalize data
train_data = utils.normalize(train_data, axis=1)
test_data = utils.normalize(test_data, axis=1)
# Build model
model = create_transformer_model()
checkpoint = keras.callbacks.ModelCheckpoint("encodedPoseEMOT_FULL.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    # TODO: Add hyperparameter grid search? definitly might be worth it for easy testing of multiple hypers.

# Train model
batch_size = 2
model.fit(train_data, train_labels, validation_data = (test_data,test_labels), epochs=20, batch_size=batch_size, callbacks=[checkpoint])




import tensorflow as tf
import numpy as np
import pickle
import os
import pandas as pd
from keras.layers import Conv1D, Add, Input, Flatten, Dense
from keras.models import Model
from keras import backend as K
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras_tuner as kt
# import tqdm

DATA_PATH = r"\Users\Lejett\Desktop\CSCE 873\taylor_datasets\NumpyFiles"
ENCODER_PATH = r"\Users\Lejett\Desktop\REPOSITORIES\FORK_EMOT\PoseEMOT\GEMEP Classification\affect_encoder_l128.h5"
SAVE_DIR_PATH = r"\Users\Lejett\Desktop\REPOSITORIES\PoseEMOT\training_output"

# SEQUENTIAL SET [dataset1:dataset2]
train_x = np.load(os.path.join(DATA_PATH, "train_x.npy"))
train_y = np.load(os.path.join(DATA_PATH, "one_hot_labels.npy"))
# SHUFFLED SET [mixed dataset1 and dataset2]
indices = np.arange(train_x.shape[0])
np.random.shuffle(indices)
shuf_train_x = train_x[indices]
shuf_train_y = train_y[indices]

# Splitting the dataset into train, validation, and test sets
shuf_train_x, shuf_test_x, shuf_train_y, shuf_test_y = train_test_split(shuf_train_x, shuf_train_y, test_size=0.3, random_state=42)
shuf_val_x, shuf_test_x, shuf_val_y, shuf_test_y = train_test_split(shuf_test_x, shuf_test_y, test_size=0.5, random_state=42)
train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.3, random_state=42)
val_x, test_x, val_y, test_y = train_test_split(test_x, test_y, test_size=0.5, random_state=42)

# Keras Tuner Hyperparameter Optimization
def build_model(hp):
    model = tf.keras.Sequential()
    # load model
    encoder = tf.keras.models.load_model(ENCODER_PATH)
    # # freeze encoder layers
    for layer in encoder.layers:
        layer.trainable = False

    for i in range(hp.Int('num_layers', 2, 5)):  # Number of layers
        model.add(tf.keras.layers.Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),
                                       activation='relu'))
        model.add(tf.keras.layers.Dropout(rate=hp.Float('dropout_' + str(i), min_value=0.1, max_value=0.5, step=0.1)))
    model.add(tf.keras.layers.Dense(7, activation='softmax'))  # Number of classes

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

shuf_tuner = kt.Hyperband(build_model,
                     objective='val_accuracy',
                     max_epochs=10,
                     directory= SAVE_DIR_PATH,
                     project_name='shuffled_emotion_classification')

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7)
shuf_tuner.search(shuf_train_x, shuf_train_y, epochs=50, validation_data=(shuf_val_x, shuf_val_y), callbacks=[early_stopping])

# Get the optimal hyperparameters
best_shuf_hps = shuf_tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the model with the best hyperparameters and train it
best_model = build_model(best_shuf_hps)
best_model.fit(shuf_train_x, shuf_train_y, epochs=100, validation_data=(shuf_val_x, shuf_val_y), callbacks=[early_stopping])

seq_tuner = kt.Hyperband(build_model,
                     objective='val_accuracy',
                     max_epochs=10,
                     directory= SAVE_DIR_PATH,
                     project_name='sequential_emotion_classification')
early_stopping2 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7)
seq_tuner.search(train_x, train_y, epochs=50, validation_data=(val_x, val_y), callbacks=[early_stopping2])
best_hps_sequential = seq_tuner.get_best_hyperparameters(num_trials=1)[0]

# Build and train the model with the best hyperparameters from the sequential tuner
best_model_sequential = build_model(best_hps_sequential)
best_model_sequential.fit(train_x, train_y, epochs=100, validation_data=(val_x, val_y), callbacks=[early_stopping2])

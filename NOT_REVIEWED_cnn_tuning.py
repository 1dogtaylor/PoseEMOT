import os
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

# Paths
DATA_PATH = r"C:\Users\Lejett\Desktop\CSCE 873\taylor_datasets\NumpyFiles"
SAVE_DIR_PATH = r"C:\Users\Lejett\Desktop\REPOSITORIES\PoseEMOT\training_output_cnn"

# Data loading and preprocessing
def load_and_preprocess_data():
    train_x = np.load(os.path.join(DATA_PATH, "train_x.npy"))
    train_y = np.load(os.path.join(DATA_PATH, "one_hot_labels.npy"))

    indices = np.arange(train_x.shape[0])
    np.random.shuffle(indices)
    shuf_train_x, shuf_train_y = train_x[indices], train_y[indices]

    return train_test_split(shuf_train_x, shuf_train_y, test_size=0.3, random_state=42)

# CNN Model building function
def build_cnn_model(hp):
    model = Sequential()

    for i in range(hp.Int('num_conv_layers', 1, 3)):
        model.add(Conv1D(filters=hp.Int(f'filters_{i}', 32, 256, step=32),
                         kernel_size=hp.Int(f'kernel_size_{i}', 3, 7),
                         activation='relu',
                         input_shape=(train_x.shape[1], train_x.shape[2]) if i == 0 else None))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(rate=hp.Float('dropout_conv_' + str(i), min_value=0.1, max_value=0.5, step=0.1)))

    model.add(Flatten())
    model.add(Dense(units=hp.Int('dense_units', 32, 256, step=32), activation='relu'))
    model.add(Dropout(rate=hp.Float('dropout_dense', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(train_y.shape[1], activation='softmax'))  # Output layer

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load and preprocess data
train_x, val_x, train_y, val_y = load_and_preprocess_data()

# Hyperparameter Tuner setup
tuner = kt.Hyperband(build_cnn_model,
                     objective='val_accuracy',
                     max_epochs=10,
                     directory=SAVE_DIR_PATH,
                     project_name='cnn_emotion_classification')

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Search for the best hyperparameters
tuner.search(train_x, train_y, epochs=50, validation_data=(val_x, val_y), callbacks=[early_stopping])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build and train the model with the best hyperparameters
best_model = build_cnn_model(best_hps)
best_model.fit(train_x, train_y, epochs=50, validation_data=(val_x, val_y), callbacks=[early_stopping])

# Save the best model
best_model.save(os.path.join(SAVE_DIR_PATH, 'best_cnn_model.h5'))

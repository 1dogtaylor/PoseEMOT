import os
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Paths
DATA_PATH = "path_to_your_data"  # Replace with your data path
LSTM_MODEL_PATH = "path_to_saved_lstm_model"  # Replace with saved LSTM model path
BILSTM_MODEL_PATH = "path_to_saved_bilstm_model"  # Replace with saved BiLSTM model path
SAVE_DIR_PATH = "path_to_save_results"  # Replace with your save directory path

def load_and_preprocess_data():
    train_x = np.load(os.path.join(DATA_PATH, "train_x.npy"))
    train_y = np.load(os.path.join(DATA_PATH, "one_hot_labels.npy"))

    indices = np.arange(train_x.shape[0])
    np.random.shuffle(indices)
    shuf_train_x, shuf_train_y = train_x[indices], train_y[indices]

    # Splitting the data into training, validation, and test sets
    train_x, test_val_x, train_y, test_val_y = train_test_split(shuf_train_x, shuf_train_y, test_size=0.3, random_state=42)
    val_x, test_x, val_y, test_y = train_test_split(test_val_x, test_val_y, test_size=0.5, random_state=42)

    return train_x, val_x, test_x, train_y, val_y, test_y

# Load pre-trained models
lstm_model = load_model(LSTM_MODEL_PATH)
bilstm_model = load_model(BILSTM_MODEL_PATH)

# Hypermodel function
def build_hypermodel(model_type, hp):
    if model_type == 'LSTM':
        model = lstm_model
    elif model_type == 'BiLSTM':
        model = bilstm_model
    else:
        raise ValueError('Invalid model type')

    # Tuning batch size
    batch_size = hp.Choice('batch_size', values=[32, 64, 128, 256])
    return model, batch_size

# Load data
train_x, val_x, test_x, train_y, val_y, test_y = load_and_preprocess_data()

# Tuner setup for LSTM
lstm_tuner = kt.Hyperband(lambda hp: build_hypermodel('LSTM', hp),
                          objective='val_accuracy',
                          max_epochs=10,
                          hyperband_iterations=2,
                          directory=SAVE_DIR_PATH,
                          project_name='lstm_batch_size_tuning')

# Tuner setup for BiLSTM
bilstm_tuner = kt.Hyperband(lambda hp: build_hypermodel('BiLSTM', hp),
                            objective='val_accuracy',
                            max_epochs=10,
                            hyperband_iterations=2,
                            directory=SAVE_DIR_PATH,
                            project_name='bilstm_batch_size_tuning')

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)


# Tuning batch sizes for LSTM
lstm_tuner.search(train_x, train_y, epochs=50, validation_data=(val_x, val_y), callbacks=[early_stopping])

# Tuning batch sizes for BiLSTM
bilstm_tuner.search(train_x, train_y, epochs=50, validation_data=(val_x, val_y), callbacks=[early_stopping])

# Extract best batch sizes and retrain models
best_lstm_batch_size = lstm_tuner.get_best_hyperparameters()[0].get('batch_size')
best_bilstm_batch_size = bilstm_tuner.get_best_hyperparameters()[0].get('batch_size')

# Retrain models with best batch sizes
lstm_model.fit(train_x, train_y, batch_size=best_lstm_batch_size, epochs=50, validation_data=(val_x, val_y), callbacks=[early_stopping])
bilstm_model.fit(train_x, train_y, batch_size=best_bilstm_batch_size, epochs=50, validation_data=(val_x, val_y), callbacks=[early_stopping])

# Save the results
lstm_model.save(os.path.join(SAVE_DIR_PATH, 'lstm_model_best_batch_size.h5'))
bilstm_model.save(os.path.join(SAVE_DIR_PATH, 'bilstm_model_best_batch_size.h5'))

# Evaluate the model on the test set
test_loss, test_accuracy = lstm_model.evaluate(test_x, test_y)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
# Evaluate the model on the test set
test_loss, test_accuracy = bilstm_model.evaluate(test_x, test_y)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

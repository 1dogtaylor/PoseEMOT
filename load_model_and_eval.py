import os
import numpy as np
import tensorflow as tf
import keras_tuner as kt
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint, Callback
from keras.models import load_model

DATA_PATH = r"C:\Users\Lejett\Desktop\CSCE 873\taylor_datasets\NumpyFiles"
SAVE_DIR_PATH = r"C:\Users\Lejett\Desktop\REPOSITORIES\PoseEMOT\hypermodel_tunes"

# Data loading and preprocessing

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

def save_summary(model, history, tuner, mode):
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{mode} Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{mode} Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')

    plt.savefig(os.path.join(SAVE_DIR_PATH, f'{mode}_training_history.png'))
    plt.close()

    with open(os.path.join(SAVE_DIR_PATH, 'summary.txt'), 'a') as f:
        f.write(f"\n{mode} Model Summary:\n")
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        f.write(f"\n{mode} Training History:\n")
        for key in history.history.keys():
            f.write(f"{key}: {history.history[key]}\n")

        f.write(f"\n{mode} Tuning Information:\n")
        for trial in tuner.oracle.get_best_trials(num_trials=1):
            f.write(f"Hyperparameters: {trial.hyperparameters.values}\n")
            f.write(f"Score: {trial.score}\n")

def save_learning_curves(tuner):
    for trial in tuner.oracle.get_trials():
        trial_id = trial.trial_id
        history = trial.metrics.get_history('accuracy')
        
        plt.figure(figsize=(12, 4))
        plt.plot([epoch['value'] for epoch in history])
        plt.title(f"Learning Curve for Trial {trial_id}")
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.savefig(os.path.join(SAVE_DIR_PATH, f'trial_{trial_id}_learning_curve.png'))
        plt.close()

train_x, val_x, test_x, train_y, val_y, test_y = load_and_preprocess_data()
best_model= load_model(os.path.join(SAVE_DIR_PATH, "best_model_checkpoint.h5"))
eval_hist = best_model.evaluate(test_x, test_y)
print(eval_hist)

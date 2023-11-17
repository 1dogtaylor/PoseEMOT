import os
import numpy as np
import tensorflow as tf
import keras_tuner as kt
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.callbacks import EarlyStopping
# from kerastuner.tuners import Hyperband
from sklearn.model_selection import train_test_split
from tqdm.keras import TqdmCallback

# Paths
DATA_PATH = r"C:\Users\Lejett\Desktop\CSCE 873\taylor_datasets\NumpyFiles"
ENCODER_PATH = r"C:\Users\Lejett\Desktop\REPOSITORIES\FORK_EMOT\PoseEMOT\GEMEP Classification\affect_encoder_l128.h5"
SAVE_DIR_PATH = r"C:\Users\Lejett\Desktop\REPOSITORIES\PoseEMOT\training_output"


# Data loading and preprocessing
def load_and_preprocess_data():
    train_x = np.load(os.path.join(DATA_PATH, "train_x.npy"))
    train_y = np.load(os.path.join(DATA_PATH, "one_hot_labels.npy"))

    indices = np.arange(train_x.shape[0])
    np.random.shuffle(indices)
    shuf_train_x, shuf_train_y = train_x[indices], train_y[indices]

    # print("Train X shape:", train_x.shape)
    # print("Train Y shape:", train_y.shape)
    # print("Shuff X shape:", shuf_train_x.shape)
    # print("Shuff Y shape:", shuf_train_y.shape)


    return train_test_split(train_x, train_y, test_size=0.3, random_state=42), \
           train_test_split(shuf_train_x, shuf_train_y, test_size=0.3, random_state=42)

# Model building
def build_model(hp):
    model = tf.keras.Sequential()
    encoder = load_model(ENCODER_PATH)
    for layer in encoder.layers:
        layer.trainable = False
    model.add(tf.keras.layers.Flatten())

    for i in range(hp.Int('num_layers', 2, 5)):
        model.add(tf.keras.layers.Dense(units=hp.Int('units_' + str(i), 32, 512, 32), activation='relu'))
        model.add(tf.keras.layers.Dropout(hp.Float('dropout_' + str(i), 0.1, 0.5, 0.1)))

    model.add(tf.keras.layers.Dense(7, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Hyperparameter tuning and training
def tune_and_train(train_x, train_y, val_x, val_y, mode):
    tuner = kt.tuners.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=10,
        directory=SAVE_DIR_PATH,
        project_name=f'{mode}_emotion_classification'
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=7)
    tuner.search(train_x, train_y, epochs=50, validation_data=(val_x, val_y), callbacks=[early_stopping], verbose=1)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = build_model(best_hps)
    history = model.fit(train_x, train_y, epochs=100, validation_data=(val_x, val_y), callbacks=[TqdmCallback(verbose=1), early_stopping])
    model.save(os.path.join(SAVE_DIR_PATH, f'{mode}_best_dense.h5'))

    return model, history, tuner

# Save model summaries and tuning information
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

    plt.savefig(f'{mode}_training_history.png')
    plt.close()

    with open('summary.txt', 'a') as f:
        f.write(f"\n{mode} Model Summary:\n")
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        f.write(f"\n{mode} Training History:\n")
        for key in history.history.keys():
            f.write(f"{key}: {history.history[key]}\n")

        f.write(f"\n{mode} Tuning Information:\n")
        for trial in tuner.oracle.get_best_trials(num_trials=1):
            f.write(f"Hyperparameters: {trial.hyperparameters.values}\n")
            f.write(f"Score: {trial.score}\n")
    

# Main execution
if __name__ == "__main__":
    (train_x, val_x, train_y, val_y), (shuf_train_x, shuf_val_x, shuf_train_y, shuf_val_y) = load_and_preprocess_data()

    # print("Train X shape:", train_x.shape)
    # print("Train Y shape:", train_y.shape)
    # print("Validation X shape:", val_x.shape)
    # print("Validation Y shape:", val_y.shape)


    seq_model, seq_history, seq_tuner = tune_and_train(train_x, train_y, val_x, val_y, "sequential")
    shuf_model, shuf_history, shuf_tuner = tune_and_train(shuf_train_x, shuf_train_y, shuf_val_x, shuf_val_y, "shuffled")

    save_summary(seq_model, seq_history, seq_tuner, "Sequential")
    save_summary(shuf_model, shuf_history, shuf_tuner, "Shuffled")

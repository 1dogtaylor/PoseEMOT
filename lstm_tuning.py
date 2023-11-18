import os
import numpy as np
import tensorflow as tf
import keras_tuner as kt
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tqdm.keras import TqdmCallback

# Paths
DATA_PATH = r"C:\Users\Lejett\Desktop\CSCE 873\taylor_datasets\NumpyFiles"
SAVE_DIR_PATH = r"C:\Users\Lejett\Desktop\REPOSITORIES\PoseEMOT\training_output_lstm"

# Data loading and preprocessing
def load_and_preprocess_data():
    train_x = np.load(os.path.join(DATA_PATH, "train_x.npy"))
    train_y = np.load(os.path.join(DATA_PATH, "one_hot_labels.npy"))

    indices = np.arange(train_x.shape[0])
    np.random.shuffle(indices)
    shuf_train_x, shuf_train_y = train_x[indices], train_y[indices]

    return train_test_split(train_x, train_y, test_size=0.3, random_state=42), \
           train_test_split(shuf_train_x, shuf_train_y, test_size=0.3, random_state=42)

# Model building
def build_model(hp):
    model = tf.keras.Sequential()
    batch_size = hp.Int('batch_size', 16, 128, step=16)

    for i in range(hp.Int('num_lstm_layers', 1, 4)):
        model.add(tf.keras.layers.LSTM(units=hp.Int('lstm_units_' + str(i), 32, 256, step=32),
                                       return_sequences=True if i < hp.get('num_lstm_layers') - 1 else False,
                                       dropout=hp.Float('lstm_dropout_' + str(i), 0.1, 0.5, step=0.1),
                                       recurrent_dropout=hp.Float('recurrent_dropout_' + str(i), 0.1, 0.5, step=0.1)))

    model.add(tf.keras.layers.Dense(units=hp.Int('dense_units', 32, 256, step=32),
                                    activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l1_l2(hp.Float('l1_reg', 1e-5, 1e-2, sampling='LOG'),
                                                                          hp.Float('l2_reg', 1e-5, 1e-2, sampling='LOG'))))

    model.add(tf.keras.layers.Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1)))
    model.add(tf.keras.layers.Dense(7, activation='softmax'))

    model.compile(optimizer=hp.Choice('optimizer', ['adam', 'rmsprop']),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG'),
                  batch_size=batch_size)

    return model

# Hyperparameter tuning and training
def tune_and_train(train_x, train_y, val_x, val_y, mode):
    tuner = kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=10,
        executions_per_trial=2,
        directory=SAVE_DIR_PATH,
        project_name=f'{mode}_emotion_classification',
        overwrite=True
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=15)
    tuner.search(train_x, train_y, epochs=50, validation_data=(val_x, val_y), callbacks=[early_stopping], verbose=1)


    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_batch_size = best_hps.get('batch_size')
    model = build_model(best_hps)
    history = model.fit(train_x, train_y, epochs=500, batch_size=best_batch_size, validation_data=(val_x, val_y), callbacks=[TqdmCallback(verbose=1), early_stopping])
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

    with open('lstm_summary.txt', 'a') as f:
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
    _, (shuf_train_x, shuf_val_x, shuf_train_y, shuf_val_y) = load_and_preprocess_data()

    shuf_model, shuf_history, shuf_tuner = tune_and_train(shuf_train_x, shuf_train_y, shuf_val_x, shuf_val_y, "lstm_shuffled")

    save_summary(shuf_model, shuf_history, shuf_tuner, "LSTM Shuffled")

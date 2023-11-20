import os
import numpy as np
import tensorflow as tf
import keras_tuner as kt
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


# Paths
# Paths
DATA_PATH = r"C:\Users\Lejett\Desktop\CSCE 873\taylor_datasets\NumpyFiles"
SAVE_DIR_PATH = r"C:\Users\Lejett\Desktop\REPOSITORIES\PoseEMOT\training_output_propersplit"

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
# BiLSTM Model building function
def build_bilstm_model(hp):
    model = Sequential()

    for i in range(hp.Int('num_bilstm_layers', 1, 3)):
        model.add(Bidirectional(LSTM(units=hp.Int('lstm_units_' + str(i), 32, 256, step=32), 
                                     return_sequences=True if i < hp.get('num_bilstm_layers') - 1 else False))) #,
                                    # input_shape=(train_x.shape[1], train_x.shape[2]) if i == 0 else None))

    model.add(Dense(units=hp.Int('dense_units', 32, 256, step=32), activation='relu'))
    model.add(Dropout(rate=hp.Float('dropout_dense', 0.1, 0.5, step=0.1)))
    model.add(Dense(train_y.shape[1], activation='softmax'))  # Output layer

    optimizer_name = hp.Choice('optimizer', ['adam', 'rmsprop'])
    learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')

    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

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

# Load and preprocess data
train_x, val_x, test_x, train_y, val_y, test_y = load_and_preprocess_data()

# Hyperparameter Tuner setup
tuner = kt.Hyperband(build_bilstm_model,
                     objective='val_accuracy',
                     max_epochs=10,
                     executions_per_trial=2,
                     directory=SAVE_DIR_PATH,
                     project_name='bilstm_2_emotion_classification')

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=0.00001)

# Search for the best hyperparameters
tuner.search(train_x, train_y, epochs=50, validation_data=(val_x, val_y), callbacks=[early_stopping, reduce_lr])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]


# Build and train the model with the best hyperparameters
best_model = build_bilstm_model(best_hps)
history = best_model.fit(train_x, train_y, epochs=200, validation_data=(val_x, val_y), callbacks=[early_stopping, reduce_lr])

# Evaluate the model on the test set
test_loss, test_accuracy = best_model.evaluate(test_x, test_y)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Save model summaries and tuning information
save_summary(best_model, history, tuner, 'bilstm_2')

# Save the best model
best_model.save(os.path.join(SAVE_DIR_PATH, 'best_bilstm_2_model.h5'))


import os
import numpy as np
import tensorflow as tf
import keras_tuner as kt
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint, Callback

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
    

class EmotionalClassificationHyperModel(kt.HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = Sequential()
        
        # RNN Layer Configuration
        for i in range(hp.Int('num_layers', 1, 3)):
            rnn_type = hp.Choice('rnn_type_' + str(i), ['LSTM', 'BiLSTM'])
            units = hp.Int('units_' + str(i), min_value=32, max_value=512, step=32)
            return_sequences = i < hp.get('num_layers') - 1
            if i == 0:
                if rnn_type == 'LSTM':
                    layer = LSTM(units, return_sequences=return_sequences, input_shape=self.input_shape)
                else:
                    layer = Bidirectional(LSTM(units, return_sequences=return_sequences, input_shape=self.input_shape))
            else:
                if rnn_type == 'LSTM':
                    layer = LSTM(units, return_sequences=return_sequences)
                else:
                    layer = Bidirectional(LSTM(units, return_sequences=return_sequences))

            model.add(layer)
            model.add(Dropout(rate=hp.Float('dropout_rnn_' + str(i), min_value=0.0, max_value=0.5, step=0.1)))

        # Dense Layer Configuration
        for j in range(hp.Int('num_dense_layers', 1, 3)):
            model.add(Dense(units=hp.Int('dense_units_' + str(j), min_value=32, max_value=256, step=32),
                            activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l1_l2(
                                hp.Float('l1_reg_' + str(j), 1e-5, 1e-2, sampling='LOG'),
                                hp.Float('l2_reg_' + str(j), 1e-5, 1e-2, sampling='LOG'))))
            model.add(Dropout(rate=hp.Float('dropout_dense_' + str(j), 0.1, 0.5, step=0.1)))

        # Output Layer
        model.add(Dense(7, activation='softmax'))

        # Optimizer
        # Optimizer Configuration
        optimizer_name = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd', 'nadam'])
        learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')

        if optimizer_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=hp.Float('momentum', 0.0, 0.9, step=0.1))
        elif optimizer_name == 'nadam':
            optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)


        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])

        return model
    
class BatchSizeCallback(Callback):
    def on_trial_begin(self, trial):
        self.batch_size = trial.hyperparameters.get('batch_size')

model_checkpoint = ModelCheckpoint(
    filepath=os.path.join(SAVE_DIR_PATH, 'best_model_checkpoint.h5'), 
    save_best_only=True, 
    monitor='val_accuracy', 
    mode='max',
    verbose=1
)

batch_size_callback = BatchSizeCallback()
    
# Load and preprocess data
train_x, val_x, test_x, train_y, val_y, test_y = load_and_preprocess_data()

# Setup the Hyperband tuner
tuner = kt.Hyperband(
    EmotionalClassificationHyperModel(input_shape=train_x.shape[1:]),
    objective='val_accuracy',
    max_epochs=10,
    hyperband_iterations=2,
    directory=SAVE_DIR_PATH,
    project_name='emotion_classification'
)

# Create callback
tensorboard = TensorBoard(log_dir=os.path.join(SAVE_DIR_PATH, 'tensorboard_logs'), histogram_freq=10)
early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.0001)

# Start the search
tuner.search(train_x, train_y, epochs=10, validation_data=(val_x, val_y), callbacks=[early_stopping, reduce_lr, batch_size_callback]) 
batch_size = batch_size_callback.batch_size

# After the search is complete
best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_models=1)[0]
best_batch_size = best_hyperparameters.get('batch_size')

# Save summary and stats
history = best_model.fit(train_x, train_y, epochs=100, validation_data=(val_x, val_y), batch_size=best_batch_size, callbacks=[tensorboard, early_stopping, reduce_lr, model_checkpoint])  # Adjust epochs as needed
save_summary(best_model, history, tuner, 'BestModel')

eval_hist = best_model.evaluate(test_x, test_y, batch_size=best_batch_size)

# Save evaluation results
with open(os.path.join(SAVE_DIR_PATH, 'summary.txt'), 'a') as f:
    f.write(f"\n\n\tFinal Test Loss: {eval_hist[0]}\n")
    f.write(f"\tTest Accuracy: {eval_hist[1]}\n")
    # Write hyperparameters
    f.write(f"\nBest Hyperparameters:\n")
    for key in best_hyperparameters.values.keys():
        f.write(f"{key}: {best_hyperparameters.values[key]}\n")
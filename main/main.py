import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, classification_report, confusion_matrix
from itertools import cycle
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback

DATA_PATH = r"C:\Users\Lejett\Desktop\CSCE 873\taylor_datasets\NumpyFiles"
ENCODER_PATH = r"C:\Users\Lejett\Desktop\REPOSITORIES\FORK_EMOT\PoseEMOT\GEMEP Classification\affect_encoder_l128.h5"
SAVED_LSTM = r"C:\Users\Lejett\Desktop\REPOSITORIES\PoseEMOT\training_output_lstm_rlr\rlr_lstm_best.h5"
SAVED_BILSTM = r"C:\Users\Lejett\Desktop\REPOSITORIES\PoseEMOT\training_output_lstm_rlr\best_bilstm_2_model.h5"
SAVED_HYPERMODEL = r"C:\Users\Lejett\Desktop\REPOSITORIES\PoseEMOT\hypermodel_tunes\best_model_checkpoint.h5"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_SPLITS_PATH = r"C:\Users\Lejett\Desktop\REPOSITORIES\PoseEMOT\main\numpy_split_saves"
SAVE_DIR_PATH = r"C:\Users\Lejett\Desktop\REPOSITORIES\PoseEMOT\main\main_output"
SAVE_FIGS_PATH = r"C:\Users\Lejett\Desktop\REPOSITORIES\PoseEMOT\main\main_output\figures"
SAVE_MODELS_PATH = r"C:\Users\Lejett\Desktop\REPOSITORIES\PoseEMOT\main\main_output\models"

def load_and_preprocess_data(normalize=True):
    # one data entry will be an emotion label for a sequence of 10 frames, 33 landmarks in each frame (each has x, y and z for 99 features)
    train_x = np.load(os.path.join(DATA_PATH, "train_x.npy"))
    train_y = np.load(os.path.join(DATA_PATH, "one_hot_labels.npy"))
    if normalize:
        train_x = train_x / np.max(train_x)

    indices = np.arange(train_x.shape[0])
    np.random.shuffle(indices)
    shuf_train_x, shuf_train_y = train_x[indices], train_y[indices]

    # Splitting the data into training, validation, and test sets
    train_x, test_val_x, train_y, test_val_y = train_test_split(shuf_train_x, shuf_train_y, test_size=0.3)
    val_x, test_x, val_y, test_y = train_test_split(test_val_x, test_val_y, test_size=0.5)

    np.save(os.path.join(SAVE_SPLITS_PATH, "train_x.npy"), train_x)
    np.save(os.path.join(SAVE_SPLITS_PATH, "val_x.npy"), val_x)
    np.save(os.path.join(SAVE_SPLITS_PATH, "test_x.npy"), test_x)
    np.save(os.path.join(SAVE_SPLITS_PATH, "train_y.npy"), train_y)
    np.save(os.path.join(SAVE_SPLITS_PATH, "val_y.npy"), val_y)
    np.save(os.path.join(SAVE_SPLITS_PATH, "test_y.npy"), test_y)

    return train_x, val_x, test_x, train_y, val_y, test_y

def save_summary(model, history,  mode):
    save_model_info(model, history, mode)
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

def save_model_info(model, history, model_name):
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    model_summary_str = "\n".join(model_summary)
    print_to_overview(f"Model Summary for {model_name}:\n{model_summary_str}")
    model_info = {
        'model_name': model_name,
        'model_summary': model_summary,
        'history': history.history if history else None
    }
    with open(os.path.join(SAVE_DIR_PATH, f'{model_name}_info.json'), 'w') as f:
        json.dump(model_info, f)

def print_to_overview(*args, **kwargs):
    with open(os.path.join(SAVE_DIR_PATH, 'overview.txt'), 'a') as f:
        print(*args, **kwargs, file=f)

def train_model(model, train_x, train_y, val_x, val_y):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=70, callbacks=[early_stopping, reduce_lr, custom_checkpoint])
    return model, history

def evaluate_model(model, test_x, test_y, model_name):
    test_loss, test_acc = model.evaluate(test_x, test_y)
    print_to_overview(f'Evaluation of {model_name}: Test Accuracy: {test_acc}, Test Loss: {test_loss}')
    predict_metrics_and_visualization(model, test_x, test_y, model_name)

def build_dense():
    model = Sequential()
    encoder = load_model(ENCODER_PATH)
    for layer in encoder.layers:
        layer.trainable = False
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(units=64, activation='relu'))
    model.add(tf.keras.layers.Dense(7, activation='softmax'))
    return model 

def build_lstm():
    model = Sequential()
    model.add(tf.keras.layers.LSTM(units=64, return_sequences=True))
    model.add(tf.keras.layers.Dropout(.4))
    model.add(tf.keras.layers.Dense(units=32, 
                                    activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l1_l2()))
    model.add(tf.keras.layers.Dense(7, activation='softmax'))
    return model

def predict_metrics_and_visualization(model, test_x, test_y, model_name, n_classes=7):
    predictions = model.predict(test_x)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(test_y, axis=1)

    # Classification report, confusion matrix, ROC curve, precision-recall curve
    print(classification_report(true_classes, predicted_classes))
    print_to_overview(classification_report(true_classes, predicted_classes))
    cm = confusion_matrix(true_classes, predicted_classes)
    plot_confusion_matrix(model_name, cm)
    plot_roc_curve(model_name, test_y, predictions, n_classes)
    plot_precision_recall_curve(model_name, test_y, predictions, n_classes)

def plot_confusion_matrix(name, cm):
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title('Confusion matrix ')
    plt.colorbar()
    plt.savefig(os.path.join(SAVE_DIR_PATH, f'{name}_confusion_matrix.png'))
    plt.close()

def plot_roc_curve(name, test_y, predictions, n_classes):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(test_y[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure()
    colors = cycle(['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(SAVE_DIR_PATH, f'{name}_roc_curve.png'))
    plt.close()

def plot_precision_recall_curve(name, test_y, predictions, n_classes):
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(test_y[:, i],
                                                            predictions[:, i])
        average_precision[i] = average_precision_score(test_y[:, i], predictions[:, i])

    # Plot all precision-recall curves
    plt.figure()
    colors = cycle(['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label='Precision-Recall curve of class {0} (AP = {1:0.2f})'
                 ''.format(i, average_precision[i]))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Multi-class Precision-Recall curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(SAVE_DIR_PATH, f'{name}precision_recall_curve.png'))
    plt.close()


def plot_multiple_history(histories, model_names):
    plt.figure(figsize=(12, 6))

    colors = cycle(['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange'])
    line_styles = ['-', '--', '-.', ':']

    for history, model_name in zip(histories, model_names):
        color = next(colors)
        for metric in ['accuracy', 'val_accuracy']:
            linestyle = '-' if 'val' not in metric else '--'
            plt.plot(history.history[metric], linestyle=linestyle, color=color)
            color = next(colors)  # Change color for validation line

    # Create legend entries
    legend_entries = []
    for model_name in model_names:
        legend_entries.append(f'{model_name} Train')
        legend_entries.append(f'{model_name} Val')

    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(legend_entries, loc='upper left')
    plt.savefig(os.path.join(SAVE_FIGS_PATH, 'model_accuracy_comparison.png'))
    plt.close()



class CustomModelCheckpoint(Callback):
    def __init__(self, filepath, save_freq=5, min_accuracy=None):
        super(CustomModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.save_freq = save_freq
        self.min_accuracy = min_accuracy

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if epoch % self.save_freq == 0 and (self.min_accuracy is None or logs.get('val_accuracy', 0) > self.min_accuracy):
            self.model.save(self.filepath.format(epoch=epoch, **logs))

#######################################################################

# load data
train_x, val_x, test_x, train_y, val_y, test_y = load_and_preprocess_data()

# callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)
custom_checkpoint = CustomModelCheckpoint(
    filepath=os.path.join(SAVE_MODELS_PATH, 'model_epoch_{epoch}.h5'),
    save_freq=5,
    min_accuracy=0.50
)

# Train and Evaluate Dense Model
print_to_overview('Dense Model:\n')
dense_m = build_dense()
dense_m, dense_history = train_model(dense_m, train_x, train_y, val_x, val_y)
evaluate_model(dense_m, test_x, test_y, 'dense')
save_summary(dense_m, dense_history, 'dense')

# Train and Evaluate LSTM Model
print_to_overview('LSTM Model:\n')
lstm_m = build_lstm()
lstm_m, lstm_history = train_model(lstm_m, train_x, train_y, val_x, val_y)
evaluate_model(lstm_m, test_x, test_y)
save_summary(lstm_m, lstm_history, 'lstm')

# plot both models training history on the same plot
plot_multiple_history([dense_history, lstm_history], ['dense', 'lstm'])

# load saved models
# evaluate saved models
print_to_overview('Saved Models:\n')
print_to_overview('Hypermodel:\n')
saved_hypermodel = load_model(SAVED_HYPERMODEL)
evaluate_model(saved_hypermodel, test_x, test_y)
save_summary(saved_hypermodel, None, 'saved_hypermodel')

print_to_overview('BiLSTM:\n')
saved_bilstm = load_model(SAVED_BILSTM)
evaluate_model(saved_bilstm, test_x, test_y)
save_summary(saved_bilstm, None, 'saved_bilstm')

print_to_overview('LSTM:\n')
saved_lstm = load_model(SAVED_LSTM)
evaluate_model(saved_lstm, test_x, test_y)
save_summary(saved_lstm, None, 'saved_lstm')

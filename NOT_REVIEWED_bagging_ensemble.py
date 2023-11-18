import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import load_model

# Paths
DATA_PATH = "path_to_your_data"  # Replace with your data path
LSTM_MODEL_PATH = "path_to_saved_lstm_model"  # Replace with saved LSTM model path
BILSTM_MODEL_PATH = "path_to_saved_bilstm_model"  # Replace with saved BiLSTM model path

# Load pre-trained models
lstm_model = load_model(LSTM_MODEL_PATH)
bilstm_model = load_model(BILSTM_MODEL_PATH)

# Load data
train_x, val_x, train_y, val_y = load_and_preprocess_data()

# Number of models in the ensemble
n_models = 10  # You can adjust this number

# Function to create a bagged dataset
def create_bagged_dataset(x, y, size):
    indices = np.random.choice(len(x), size=size, replace=True)
    return x[indices], y[indices]

# Train multiple models on bagged datasets
ensemble_predictions = []
for i in range(n_models):
    # Create bagged dataset
    bag_x, bag_y = create_bagged_dataset(train_x, train_y, len(train_x))

    # Choose model type for this iteration
    model = lstm_model if i % 2 == 0 else bilstm_model  # Alternate between LSTM and BiLSTM

    # Train the model
    model.fit(bag_x, bag_y, epochs=30, batch_size=64, verbose=0)

    # Predict on validation set
    ensemble_predictions.append(model.predict(val_x))

# Aggregate predictions
final_predictions = np.mean(ensemble_predictions, axis=0)
final_predictions = np.argmax(final_predictions, axis=1)

# Evaluate performance
accuracy = np.mean(final_predictions == np.argmax(val_y, axis=1))
print("Ensemble Accuracy:", accuracy)

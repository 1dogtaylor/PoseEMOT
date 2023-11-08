import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping

# Function to load data
def load_data(data_path):
    if data_path.endswith('.csv'):
        return pd.read_csv(data_path)
    elif data_path.endswith('.pkl'):
        return pd.read_pickle(data_path)
    else:
        raise ValueError("Unsupported file type. Only CSV and PKL files are supported.")

# Function to preprocess the data
def preprocess_data(df, sequence_length=10):
    frame_set = []
    emotion_set = []

    pose_frames = []
    emotions = []

    for _, row in df.iterrows():
        pose_frames.append(row['pose'])
        emotions.append(row['emotion'])
        if len(pose_frames) == sequence_length:
            frame_set.append(pose_frames.copy())
            emotion_set.append(emotions.copy())
            pose_frames.pop(0)
            emotions.pop(0)
    
    return frame_set, emotion_set

# Function to get the most common emotion in a sequence
def get_most_common_emotions(emotion_set):
    label_encoder = LabelEncoder()
    encoded_emotions = [label_encoder.fit_transform(sequence) for sequence in emotion_set]
    most_common_emotions = [np.bincount(sequence).argmax() for sequence in encoded_emotions]
    return to_categorical(most_common_emotions)

# Function to create the model
def create_model(input_shape, encoder_path):
    # Load the encoder
    try:
        encoder = load_model(encoder_path)
    except Exception as e:
        raise IOError(f"Error loading the encoder model: {e}")

    # Freeze the encoder layers
    for layer in encoder.layers:
        layer.trainable = False

    input_seq = Input(shape=input_shape)
    x = encoder(input_seq)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(7, activation='softmax')(x)

    model = Model(inputs=input_seq, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Main function to run the pipeline
def main(data_path, encoder_path, output_model_path):
    df = load_data(data_path)

    frame_set, emotion_set = preprocess_data(df)
    most_common_emotions = get_most_common_emotions(emotion_set)

    train_x = np.array(frame_set).reshape(-1, 10, 33*3)
    train_y = np.array(most_common_emotions)

    model = create_model(train_x.shape[1:], encoder_path)

    early_stopping = EarlyStopping(monitor='val_loss', patience=7)
    model.fit(train_x, train_y, epochs=800, batch_size=32, callbacks=[early_stopping])

    model.save(output_model_path)
    print(f"Model saved to {output_model_path}")

if __name__ == "__main__":
    data_path = input("Enter the path to your data file: ")
    encoder_path = 'affect_encoder_l128.h5'  # Or input("Enter the path to your encoder model: ")
    output_model_path = 'Emot_classifier_l128

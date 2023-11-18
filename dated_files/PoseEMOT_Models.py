import keras 
from keras import layers
import numpy as np
import tensorflow as tf

### HYPERPARAMETER SEARCH USAGE ###
# from sklearn.model_selection import GridSearchCV
# from keras.wrappers.scikit_learn import KerasClassifier

# lstm_model = KerasClassifier(build_fn=create_lstm_model, epochs=20, batch_size=2, verbose=0)
# grid = GridSearchCV(estimator=lstm_model, param_grid=param_grid, n_jobs=-1)
# grid_result = grid.fit(train_data, train_labels)

# ### RESULTS
# print("best result: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

param_grid = { # For hyperparameter grid search DO NOT FORGET TO INCLUDE THIS
    'units': [32, 64, 128],
    'dropout': [0.1, 0.2, 0.3]
}

def get_positional_encoding(seq_len, d_model):
    """
    Return positional encoding for a given sequence length and model depth.
    """
    # Compute the positional encodings
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return tf.constant(pe, dtype=tf.float32)

# creates variable length lstm
def create_vl_lstm_model(units=64, dropout=0.2):
    model = keras.Sequential()
    model.add(layers.Masking(mask_value=0., input_shape=(None, 19)))  # None for variable length sequences
    model.add(layers.LSTM(units))
    model.add(layers.Dense(units, activation='relu'))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(4, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# creates transformer
def create_transformer_model(d_model=64, num_heads=4, ff_dim=128, dropout=0.2):
    inputs = layers.Input(shape=(None, 19))
    
	# Add positional encodings
    seq_len = tf.shape(inputs)[1]
    pos_enc = get_positional_encoding(seq_len, d_model)
    x = inputs + pos_enc
    
    x = layers.Masking(mask_value=0.)(inputs)
    transformer_encoder = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
    x = transformer_encoder(x, x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(ff_dim, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(4, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# creates original LSTM taylor built. for historical purposes.
def create_original_lstm(max_length):
    model = keras.Sequential()
    model.add(layers.Masking(mask_value=0., input_shape=(max_length, 19))) # mask_value=0. for padding NOTE find max length and put here
    model.add(layers.LSTM(64))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(4, activation='softmax'))
    
    OPTIMIZER = keras.optimizers.Adam(lr=0.001)
    model.compile(OPTIMIZER, loss='CategoricalCrossentropy', metrics=['accuracy'])
    return model


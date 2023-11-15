import tensorflow as tf
import numpy as np
import pickle
import os
import pandas as pd
from keras.layers import Conv1D, Add, Input, Flatten, Dense
from keras.models import Model
from keras import backend as K
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import tqdm


MEEDdf = pd.DataFrame(columns=['filename','frame','pose set','emotion','pose','pictures'])
GEMEPdf = pd.DataFrame(columns=['filename','frame','pose set','emotion','pose','pictures'])

path = "/Users/taylorbrandl/Taylor/Python/Nimbus/DroneFollower/Pose Estimation/POSE DATA/MEED/"
print("Loading MEED data...")
for file in os.listdir(path):
	if file.endswith('.pkl') and file.startswith('full'):
		with open(path+file, 'rb') as f:
			data = pickle.load(f)
			MEEDdf = pd.concat([MEEDdf, data])
			print(file)


coords = []
pose_set = []
emotion_set = []
print(f"DF shape: {MEEDdf.shape}")

# for meed data bc it is not saved as a 33 point pose
pbar = tqdm.tqdm(total=len(MEEDdf))
i = 0
for idx, row in MEEDdf.iterrows():
	pbar.update(1)
	coords.append(row['pose'])
	if len(coords) == 33:
		pose_set.append(coords.copy())
		emotion_set.append(row['emotion'])
		coords = []
		i += 1
	# if i == 1000: # for testing 1000 poses
	# 	break
print(f"Number of poses: {len(pose_set)}")
print(f"Number of emotions: {len(emotion_set)}")


MEEDframe_set = []
current_frame_set = []
labels = []
i = 0

for pose in pose_set:
	i += 1
	current_frame_set.append(pose)
	if len(current_frame_set) == 11:
		current_frame_set.pop(0)
		MEEDframe_set.append(current_frame_set.copy())
		current_emotions = emotion_set[i-1] # just need one emotion per frame set bc they are all the same
		labels.append(current_emotions)
	# if i == 1000:
	# 	break

print(f"MEED Sliding frame shape: {len(MEEDframe_set)}")
print(f"Sliding emotion shape: {len(labels)}")


i = 0
MEEDone_hot_labels = []
for label in labels:
	if label == "A": # anger
		MEEDone_hot_labels.append([1,0,0,0,0,0,0])
	elif label == "D": # disgust
		MEEDone_hot_labels.append([0,1,0,0,0,0,0])
	elif label == "F": # fear
		MEEDone_hot_labels.append([0,0,1,0,0,0,0])
	elif label == "H": # happy
		MEEDone_hot_labels.append([0,0,0,1,0,0,0])
	elif label == "N": # neutral
		MEEDone_hot_labels.append([0,0,0,0,1,0,0])
	elif label == "SA": # sad
		MEEDone_hot_labels.append([0,0,0,0,0,1,0])
	elif label == "SU": # surprise
		MEEDone_hot_labels.append([0,0,0,0,0,0,1])
	else:
		# pop the current frame set and do not append the label
		MEEDframe_set.pop(i)
	i += 1


path = "/Users/taylorbrandl/Taylor/Python/Nimbus/DroneFollower/Pose Estimation/POSE DATA/GEMEP_Coreset_Full Body/"
print("Loading GEMEP data...")
for file in os.listdir(path):
	if file.endswith('.pkl') and file.startswith('full'):
		with open(path+file, 'rb') as f:
			data = pickle.load(f)
			GEMEPdf = pd.concat([GEMEPdf, data])
			print(file)



prev_emotions = []
pose_frames = []
GEMEPframe_set = []
emotion_set = []
videos = []

i = 0
for idx, row in GEMEPdf.iterrows():
	i += 1
	pose_frames.append(row['pose'])
	emotion = row['filename'].split("_")[0]
	# get the last 3 characters of the emotion
	emotion = emotion[-3:]
	
	if len(pose_frames) == 11:
		pose_frames.pop(0)
		GEMEPframe_set.append(pose_frames.copy())
		emotion_set.append(emotion)

print(f"Number of GEMEP frame sets: {len(GEMEPframe_set)}")
print(f"Number of emotions: {len(emotion_set)}")

i = 0
GEMEPone_hot_labels = []
for label in emotion_set:
	if label == "ang" or label == "irr": # anger
		GEMEPone_hot_labels.append([1,0,0,0,0,0,0])
	elif label == "dis" or label == "con": # disgust
		GEMEPone_hot_labels.append([0,1,0,0,0,0,0])
	elif label == "fea" or label == "anx": # fear
		GEMEPone_hot_labels.append([0,0,1,0,0,0,0])
	elif label == "amu" or label == "pri" or label == "joy" or label == "adm": # happy
		GEMEPone_hot_labels.append([0,0,0,1,0,0,0])
	elif label == "int" or label == "rel" or label == "ple" or label == "ten": # neutral
		GEMEPone_hot_labels.append([0,0,0,0,1,0,0])
	elif label == "des" or label == "sad": # sad
		GEMEPone_hot_labels.append([0,0,0,0,0,1,0])
	elif label == "sur": # surprise
		GEMEPone_hot_labels.append([0,0,0,0,0,0,1])
	else:
		# pop the current frame set and do not append the label
		GEMEPframe_set.pop(i)
	i += 1

# combine the two datasets
frame_set = MEEDframe_set + GEMEPframe_set
one_hot_labels = MEEDone_hot_labels + GEMEPone_hot_labels

# convert to numpy arrays
train_x = np.array(frame_set)
train_x = tf.reshape(train_x, (-1, 10, 33*3))
one_hot_labels = np.array(one_hot_labels)


print(f"Train data shape:{train_x.shape}")
print(f"Train label shape:{one_hot_labels.shape}")


# load model
encoder = tf.keras.models.load_model('affect_encoder_l128.h5')



# build sequential model
input_seq = Input(shape=(10, 33*3))
x = encoder(input_seq)
x = Dense(512, activation='relu')(x)
x = Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
output = Dense(7, activation='softmax')(x)

model = Model(inputs=input_seq, outputs=output)

# freeze encoder layers
for layer in encoder.layers:
	layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=7)
model.fit(train_x, one_hot_labels, epochs=800, callbacks=[early_stopping])

model.save('MEED_Emot_classifier2_l128.h5')
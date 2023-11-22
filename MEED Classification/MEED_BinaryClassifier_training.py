
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
import tensorflow as tf

def load_data():
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
	filename_current = ""
	filename_prev = ""
	print(f"DF shape: {MEEDdf.shape}")

	# for meed data bc it is not saved as a 33 point pose
	pbar = tqdm.tqdm(total=len(MEEDdf))
	i = 0
	for idx, row in MEEDdf.iterrows():
		pbar.update(1)
		filename_current = row['filename']
		filename_current = filename_current.split("_")[2]

		if filename_current == filename_prev: # if the filename is the same, append the pose
			coords.append(row['pose'])
		else: # if the filename changes, start a new frame set
			coords = []
			coords.append(row['pose'])

		if len(coords) == 33:
			pose_set.append(coords.copy())
			emotion_set.append(row['emotion'])
			coords = []
			i += 1
		# if i == 1000: # for testing 1000 poses
		# 	break
		filename_prev = filename_current

	print(f"Number of poses: {len(pose_set)}")
	print(f"Number of emotions: {len(emotion_set)}")

	MEEDframe_set = []
	current_frame_set = []
	labels = []
	i = 0
	current_emotion = ""
	prev_emotion = ""

	for pose in pose_set:
		current_frame_set.append(pose)
		current_emotion = emotion_set[i]

		if len(current_frame_set) == 11 and current_emotion == prev_emotion:
			current_frame_set.pop(0)
			MEEDframe_set.append(current_frame_set.copy())
			current_emotion_label = emotion_set[i-1] # just need one emotion per frame set bc they are all the same
			labels.append(current_emotion_label)

		if current_emotion != prev_emotion and i!=0: # if the emotion changes, start a new frame set
			current_frame_set = []
		prev_emotion = current_emotion
		# if i == 1000:
		# 	break
		i += 1

	print(f"MEED Sliding frame shape: {len(MEEDframe_set)}")
	print(f"Sliding emotion shape: {len(labels)}")

	path = "/Users/taylorbrandl/Taylor/Python/Nimbus/DroneFollower/Pose Estimation/POSE DATA/GEMEP_Coreset_Full Body/"
	print("Loading GEMEP data...")
	for file in os.listdir(path):
		if file.endswith('.pkl') and file.startswith('full'):
			with open(path+file, 'rb') as f:
				data = pickle.load(f)
				GEMEPdf = pd.concat([GEMEPdf, data])
				print(file)

	pose_frames = []
	GEMEPframe_set = []
	emotion_set = []
	prev_emotion = ""

	i = 0
	for idx, row in GEMEPdf.iterrows():

		pose_frames.append(row['pose'])
		emotion = row['filename'].split("_")[0]
		# get the last 3 characters of the emotion
		emotion = emotion[-3:]

		if len(pose_frames) == 11 and emotion == prev_emotion:
			pose_frames.pop(0)
			GEMEPframe_set.append(pose_frames.copy())
			emotion_set.append(emotion)

		if emotion != prev_emotion and i!=0: # if the emotion changes, start a new frame set
			pose_frames = []

		prev_emotion = emotion
		i += 1

	print(f"Number of GEMEP frame sets: {len(GEMEPframe_set)}")
	print(f"Number of emotions: {len(emotion_set)}")

	# convert to numpy arrays
	MEEDframe_set = np.array(MEEDframe_set)
	GEMEPframe_set = np.array(GEMEPframe_set)
	labels = np.array(labels)
	emotion_set = np.array(emotion_set)

	MEED_labels = meed_one_hot_labels(labels, "", "multiclass")
	GEMEP_labels = gemep_one_hot_labels(emotion_set, "", "multiclass")

	# combine the data
	X = np.concatenate((MEEDframe_set, GEMEPframe_set), axis=0)
	Y = np.concatenate((MEED_labels, GEMEP_labels), axis=0)

	# save
	np.save('MEED_GEMEP_x_7.npy', X)
	np.save('MEED_GEMEP_y_7.npy', Y)

def meed_one_hot_labels(labels,emotion_class,mode):
	if mode == "binary":
		# Map of emotion classes to their associated labels
		if emotion_class == "Angry":
			emotion_class = "A"
		elif emotion_class == "Disgust":
			emotion_class = "D"
		elif emotion_class == "Fear":
			emotion_class = "F"
		elif emotion_class == "Happy":
			emotion_class = "H"
		elif emotion_class == "Neutral":
			emotion_class = "N"
		elif emotion_class == "Sad":
			emotion_class = "SA"
		elif emotion_class == "Surprise":
			emotion_class = "SU"
		MEEDone_hot_labels = []
		for label in labels:
			if label == emotion_class:
				MEEDone_hot_labels.append([1,0])
			else:
				MEEDone_hot_labels.append([0,1])
		return MEEDone_hot_labels
	
	elif mode == "multiclass": # NOTE the emotion class is not used here
		MEEDone_hot_labels = []
		for label in labels:
			if label == "A":
				MEEDone_hot_labels.append([1,0,0,0,0,0,0])
			elif label == "D":
				MEEDone_hot_labels.append([0,1,0,0,0,0,0])
			elif label == "F":
				MEEDone_hot_labels.append([0,0,1,0,0,0,0])
			elif label == "H":
				MEEDone_hot_labels.append([0,0,0,1,0,0,0])
			elif label == "N":
				MEEDone_hot_labels.append([0,0,0,0,1,0,0])
			elif label == "SA":
				MEEDone_hot_labels.append([0,0,0,0,0,1,0])
			elif label == "SU":
				MEEDone_hot_labels.append([0,0,0,0,0,0,1])
		return MEEDone_hot_labels

def gemep_one_hot_labels(labels, emotion_class, mode):

	if mode == "binary":
		# Map of emotion classes to their associated labels
		emotion_classes = {
			"Angry": ["ang", "irr"],
			"Disgust": ["dis", "con"],
			"Fear": ["fea", "anx"],
			"Happy": ["amu", "pri", "joy", "adm"],
			"Neutral": ["int", "rel", "ple", "ten"],
			"Sad": ["des", "sad"],
			"Surprise": ["sur"]
		}


		# Get the specific labels for the emotion class
		specific_labels = emotion_classes[emotion_class]
	
		# Create one-hot encoded labels
		GEMEPone_hot_labels = []
		for label in labels:
			if label in specific_labels:
				GEMEPone_hot_labels.append([1, 0])
			else:
				GEMEPone_hot_labels.append([0, 1])
	
		return GEMEPone_hot_labels
	
	elif mode == "multiclass": # NOTE the emotion class is not used here
		GEMEPone_hot_labels = []
		for label in labels:
			if label == "ang" or label == "irr":
				GEMEPone_hot_labels.append([1,0,0,0,0,0,0])
			elif label == "dis" or label == "con":
				GEMEPone_hot_labels.append([0,1,0,0,0,0,0])
			elif label == "fea" or label == "anx":
				GEMEPone_hot_labels.append([0,0,1,0,0,0,0])
			elif label == "amu" or label == "pri" or label == "joy" or label == "adm":
				GEMEPone_hot_labels.append([0,0,0,1,0,0,0])
			elif label == "int" or label == "rel" or label == "ple" or label == "ten":
				GEMEPone_hot_labels.append([0,0,0,0,1,0,0])
			elif label == "des" or label == "sad":
				GEMEPone_hot_labels.append([0,0,0,0,0,1,0])
			elif label == "sur":
				GEMEPone_hot_labels.append([0,0,0,0,0,0,1])
		return GEMEPone_hot_labels

def classifier_training(X,Y, classifier_name, mode):
	
	# reshape the data
	train_x = tf.reshape(X, (-1, 10, 33*3))


	# shuffle the data
	indices = np.arange(train_x.shape[0])
	np.random.shuffle(indices)
	indices_tf = tf.convert_to_tensor(indices, dtype=tf.int32)
	shuf_train_x, shuf_train_y = tf.gather(train_x, indices_tf), tf.gather(Y, indices_tf)
	# split into train and test
	train_x = shuf_train_x[:int(0.8*len(shuf_train_x))]
	train_y = shuf_train_y[:int(0.8*len(shuf_train_y))]
	test_x = shuf_train_x[int(0.8*len(shuf_train_x)):]
	test_y = shuf_train_y[int(0.8*len(shuf_train_y)):]

	print(f"Train data shape:{train_x.shape}")
	print(f"Train label shape:{train_y.shape}")


	# load model
	encoder = tf.keras.models.load_model('Models/affect_encoder_l128.h5')


	if mode == "binary":
		# build sequential model
		input_seq = Input(shape=(10, 33*3))
		x = encoder(input_seq)
		x = Dense(64, activation='relu')(x)
		x = Dense(64, activation='relu')(x)
		x = tf.keras.layers.Dropout(0.2)(x)
		output = Dense(2, activation='softmax')(x)

		model = Model(inputs=input_seq, outputs=output)

	elif mode == "multiclass":
		# build sequential model
		input_seq = Input(shape=(10, 33*3))
		x = encoder(input_seq)
		x = Dense(256, activation='relu')(x)
		x = Dense(256, activation='relu')(x)
		x = tf.keras.layers.Dropout(0.2)(x)
		output = Dense(7, activation='softmax')(x)

		model = Model(inputs=input_seq, outputs=output)

	# freeze encoder layers
	for layer in encoder.layers:
		layer.trainable = False

	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	model.summary()
	early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7)
	checkpoint = tf.keras.callbacks.ModelCheckpoint(classifier_name, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	history = model.fit(train_x, train_y, validation_data=[test_x, test_y], epochs=800, callbacks=[early_stopping, checkpoint])

	#save the model and history
	with open(f"{classifier_name}_history.pkl", 'wb') as f:
		pickle.dump(history.history, f)
	model.save(classifier_name)


# load the data
#load_data()

X = np.load('MEED_GEMEP_x_7.npy')
Y = np.load('MEED_GEMEP_y_7.npy')
print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")

emotions_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
for emotion_class in emotions_classes:
	print(f"Training {emotion_class} binary classifier...")
	classifier_training(X,Y,f"MEED_GEMEP_s_binary_classifier{emotion_class}_l128.h5","binary")

# train the large classifier
print("Training large classifier...")
classifier_training(X,Y,"MEED_GEMEP_L_classifier_l128.h5","multiclass")
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def pose_extract(frame):
	with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

		results = pose.process(frame)
		if drawing:
			mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
		pose_landmarks = results.pose_landmarks
		if pose_landmarks is not None: # ensures pose is detected
			landmarks = [[lmk.x, lmk.y, lmk.z] for lmk in pose_landmarks.landmark]
			if len(landmarks) == 33:
				return landmarks
			else:
				return None
		else:
			return None





# load the model
model = tf.keras.models.load_model('MEED_Emot_classifier2_l128.h5')
#emotions = ['Happy', 'Neutral', 'Sad', 'Surprise', 'Angry', 'Fear', 'Disgust'] # GEMEP
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'] # MEED
frameset = []
drawing = True
mode = "webcam"
if mode == "webcam":
	webcam = cv2.VideoCapture(0)
	while True:
		ret, frame = webcam.read()
		if not ret:
			break

		landmarks = pose_extract(frame)
		if landmarks is not None:
			frameset.append(landmarks)
			if len(frameset) == 11:
				frameset.pop(0)
				X = np.array(frameset)
				X = tf.reshape(X, (-1, 10, 33*3))
				Y = model.predict(X)
				Yindex = np.argmax(Y)
				emotion = emotions[Yindex]
				cv2.putText(frame, emotion, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
		cv2.imshow('frame', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
elif mode == "file":
	filepath = "/Users/taylorbrandl/Taylor/Python/Nimbus/DroneFollower/Pose Estimation/POSE DATA/GEMEP_Coreset_Full Body/"
	for file in os.listdir(filepath):
		if file.endswith(".avi"):
			cap = cv2.VideoCapture(filepath+file)
			while True:
				ret, frame = cap.read()
				if not ret:
					break

				landmarks = pose_extract(frame)
				if landmarks is not None:
					frameset.append(landmarks)
					if len(frameset) == 11:
						frameset.pop(0)
						X = np.array(frameset)
						X = tf.reshape(X, (-1, 10, 33*3))
						Y = model.predict(X)
						Yindex = np.argmax(Y)
						emotion = emotions[Yindex]
						cv2.putText(frame, emotion, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
						gt = file.split("_")[0]
						gt = gt[-3:]
						cv2.putText(frame, gt, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
				cv2.imshow('frame', frame)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
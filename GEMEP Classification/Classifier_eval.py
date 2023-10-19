import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# load model
classifier = tf.keras.models.load_model('Emot_classifier_l128.h5')

mode = 'webcam'
imshow = True
console_print = True
skeleton = True
exe_time = True

def get_landmarks(frame):

	with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
		results = pose.process(frame)

		pose_landmarks = results.pose_landmarks

		if skeleton:
			mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

		if pose_landmarks is not None: # ensures pose is detected
			return [[lmk.x, lmk.y, lmk.z] for lmk in pose_landmarks.landmark], frame
		else:
			return None, frame

def get_emotion(y):
	label = np.argmax(y, axis=1)

	if label == 0:
		return "happy"
	elif label == 1:
		return "neutral"
	elif label == 2:
		return "sad"
	elif label == 3:
		return "surprise"
	elif label == 4:
		return "angry"
	elif label == 5:
		return "fear"
	elif label == 6:
		return "disgust"

if mode == 'webcam':
	cap = cv2.VideoCapture(0)

	if not cap.isOpened():
		print("Cannot open camera")
		exit()

	frame_count = 0
	pose_window = []

	while True:
		if exe_time:
			start_time = time.time()
		ret, frame = cap.read()
		if not ret:
			print("Can't receive frame. Exiting ...")
			break

		# invert webcam image
		framef = cv2.flip(frame, 1)

		# detect landmarks
		landmarks, framef = get_landmarks(framef)

		# set sliding window and predict emotion
		if landmarks is not None:
			pose_window.append(landmarks)
			if len(pose_window) == 11:
				pose_window.pop(0)
				x = np.array(pose_window)
				x = tf.reshape(x, (-1, 10, 33*3))
				y = classifier.predict(x, verbose=0)
				emot = get_emotion(y)

				if console_print:
					print(emot)

				if imshow:
					# annotate frame
					cv2.putText(framef, emot, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
		
		if imshow:
			cv2.imshow('framef', framef)

		if cv2.waitKey(1) == ord('q'):
			break

		if exe_time:
			print(f"Execution time: {time.time() - start_time}")

	cap.release()
	cv2.destroyAllWindows()
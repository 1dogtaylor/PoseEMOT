import numpy as np
import cv2
import mediapipe as mp
import os
from deepface import DeepFace
import pandas as pd
import tqdm
from video_name_processing import get_file_names_from_all_pickle, get_video_files_from_path, is_video_processed
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def crop(frame, rectangle):

	x = rectangle['x']
	y = rectangle['y']
	w = rectangle['w']
	h = rectangle['h']
	scaleh = np.int(h/2)
	scalew = np.int(w/4)
	cropxmin = x-scalew
	if cropxmin < 0:
		cropxmin = 0
	cropymin = y-scaleh
	if cropymin < 0:
		cropymin = 0
	cropxmax = x+w+scalew
	cropymax = y+h+scaleh
	
	#crop the face
	cropped = frame[cropymin:cropymax, cropxmin:cropxmax]
	return cropped

def pose_extract(frame):
	with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

		results = pose.process(frame)
		if drawing:
			mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
		pose_landmarks = results.pose_landmarks
		if pose_landmarks is not None: # ensures pose is detected
			landmarks_full = [[lmk.x, lmk.y, lmk.z] for lmk in pose_landmarks.landmark]
			landmarks_visible = [landmarks_full[i] for i, lmk in enumerate(pose_landmarks.landmark) if lmk.visibility > 0.5]
			return landmarks_full, landmarks_visible
		else:
			return None, None

def pose_index(pose,faces,frameshape, frame):
	# get mediapipe center
	mediapipe_center = (pose[0][0], pose[0][1]) # nose of the person
	# convert to pixels
	mediapipe_center = (mediapipe_center[0]*frameshape[1], mediapipe_center[1]*frameshape[0]) # convert to pixels NOTE cv2 uses (height, width) and mediapipe uses (width, height)
	# get deepface center
	deepface_centers = []
	for face in faces:
		rectangle = face['region']
		x = rectangle['x']
		y = rectangle['y']
		w = rectangle['w']
		h = rectangle['h']
		deep_face_center = (x+w/2, y+h/2)
		center = (np.abs(deep_face_center[0] - mediapipe_center[0]), np.abs(deep_face_center[1] - mediapipe_center[1]))
		center = sum(center)
		if drawing:
			# draw rectangle around face
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
			cv2.imshow('frame', frame)
			cv2.waitKey(1)
		

		deepface_centers.append(center)
	index = np.argmin(deepface_centers)
	if deepface_centers[index] < 50: # likely the same person
		return index
	else:
		return None

def frame_extract(frame):
	frameshape = frame.shape
	# get mediapipe pose
	pose_full, pose_vis = pose_extract(frame)

	# if person found get the emotion and match it to the pose found
	if pose_full != None and pose_full[0] != None: # if pose and nose is detected
		# get deepface emotion
		deepface_predict = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
		# get index of person who matches the mediapipe pose
		index = pose_index(pose_full, deepface_predict, frameshape, frame)

		
		if index != None:
			return pose_full, pose_vis, deepface_predict[index]['dominant_emotion']
		else: # if no person found by matching pose
			return None, None, None
	else: # if no person found by mediapipe
		return None, None, None

# path = "/Users/taylorbrandl/Taylor/Python/Nimbus/DroneFollower/Pose Estimation/POSE DATA/Ted Talks/"
path = "C:/Users/Lejett/Desktop/TED_Dataset"
test_path = "/Users/taylorbrandl/Taylor/Python/Nimbus/DroneFollower/Pose Estimation/POSE DATA/Ted Talks/test.mp4"
path_to_pkl_directory = "./VideoFiles"
first = True
df_full = pd.DataFrame(columns=['filename','frame','pose set','emotion','pose','pictures'])
df_vis = pd.DataFrame(columns=['filename','frame','pose_set','emotion','pose','pictures'])
df_3min = pd.DataFrame(columns=['filename','frame','pose_set','emotion','pose','pictures'])
drawing = False
video_files = [] 
# get list of files at directory
# video_files = [filename for filename in os.listdir(path) if filename.endswith(('.mp4', '.avi', '.MOV', '.mov')) and not filename.__contains__('Test') and  not filename.__contains__('2023-karen-bakker-003-398193b5-571f-435d-82e1-5000k')]
video_files = get_video_files_from_path(path)
all_processed_video_names = get_file_names_from_all_pickle(path_to_pkl_directory)
# Filter out already processed videos
video_files = [video for video in video_files if not is_video_processed(video, all_processed_video_names)]

total_videos = len(video_files)
videos_processed = 0
total_poses = 0
total_3min_poses = 0
frames_processed = 0
for filename in video_files:
	vibobj = cv2.VideoCapture(path+filename)
	#vibobj = cv2.VideoCapture(test_path)
	pbar = tqdm.tqdm(total=int(vibobj.get(cv2.CAP_PROP_FRAME_COUNT))-3)
	success = True
	poses_full = []
	poses_vis = []
	emotions = []
	vid_df_full = pd.DataFrame(columns=['filename','frame','pose set','emotion','pose','pictures'])
	vid_df_vis = pd.DataFrame(columns=['filename','frame','pose set','emotion','pose','pictures'])
	vid_df_3min = pd.DataFrame(columns=['filename','frame','pose set','emotion','pose','pictures'])
	current_pose_df = pd.DataFrame(columns=['filename','frame','pose set','emotion','pose','pictures'])
	current_pose_df_vis = pd.DataFrame(columns=['filename','frame','pose set','emotion','pose','pictures'])
	current_pose_df_3min = pd.DataFrame(columns=['filename','frame','pose set','emotion','pose','pictures'])
	pose_set = 0
	three_min_pose_set = 0
	consecutive_frames = 0
	consecutive_min_3_frames = 0
	missed_frames = 0
	toggle = False
	toggle_3min = False
	i = 0
	while success:
		pbar.update(1)
		# Read a new frame
		success, frame = vibobj.read()
		if success == False:
			continue
		frames_processed += 1
		# get the pose and corresponding emotion
		pose_full, pose_vis, emotion = frame_extract(frame)

		if pose_full != None and pose_vis != None and emotion != None: # if pose detected append to df
			consecutive_frames += 1 # count consecutive frames
			consecutive_min_3_frames += 1
			current_pose_df = current_pose_df.append({'filename': filename, 'frame': i,'pose set': pose_set, 'emotion': emotion, 'pose': pose_full, 'pictures': None}, ignore_index=True) # append to current pose df
			current_pose_df_vis = current_pose_df_vis.append({'filename': filename, 'frame': i,'pose set': pose_set, 'emotion': emotion, 'pose': pose_vis, 'pictures': None}, ignore_index=True)
			current_pose_df_3min = current_pose_df_3min.append({'filename': filename, 'frame': i,'pose set': pose_set, 'emotion': emotion, 'pose': pose_vis, 'pictures': None}, ignore_index=True)
			print(f"v:{videos_processed}/{total_videos} f:{frames_processed} p:{total_poses} p3:{total_3min_poses} cp:{pose_set} cp3:{three_min_pose_set} fc:{consecutive_frames}/60 fc3:{consecutive_min_3_frames}/60")
			
			if consecutive_frames >= 60: # if 60 consecutive frames of the same pose (2 seconds)
				if not toggle: # if this is the first frame of the pose set
					vid_df_full = vid_df_full.append(current_pose_df, ignore_index=True)
					vid_df_vis = vid_df_vis.append(current_pose_df_vis, ignore_index=True)
					current_pose_df = pd.DataFrame(columns=['filename','frame','pose set','emotion','pose','pictures'])
					current_pose_df_vis = pd.DataFrame(columns=['filename','frame','pose set','emotion','pose','pictures'])
				else: # if this is not the first frame of the pose set
					vid_df_full = vid_df_full.append({'filename': filename, 'frame': i,'pose set': pose_set, 'emotion': emotion, 'pose': pose_full, 'pictures': None}, ignore_index=True)
					vid_df_vis = vid_df_vis.append({'filename': filename, 'frame': i,'pose set': pose_set, 'emotion': emotion, 'pose': pose_vis, 'pictures': None}, ignore_index=True)
				toggle = True
			if consecutive_min_3_frames >= 60:
				if not toggle_3min:
					vid_df_3min = vid_df_3min.append(current_pose_df_3min, ignore_index=True)
					current_pose_df_3min = pd.DataFrame(columns=['filename','frame','pose set','emotion','pose','pictures'])
				else:
					vid_df_3min = vid_df_3min.append({'filename': filename, 'frame': i,'pose set': pose_set, 'emotion': emotion, 'pose': pose_vis, 'pictures': None}, ignore_index=True)
				toggle_3min = True
		else:
			if toggle: # if this is the last frame of the pose set
				pose_set += 1
				print(f"pose count:{pose_set}------------------------------------")
				toggle = False

			consecutive_frames = 0
			
			missed_frames += 1
			if missed_frames >= 3:
				consecutive_min_3_frames = 0
				missed_frames = 0
				if toggle_3min:
					three_min_pose_set += 1
					print(f"3min pose count:{three_min_pose_set}------------------------------------")
					toggle_3min = False

		# # temporary break at 200 frames for testing
		i += 1
		# if i == 2000:
		# 	break

	# append to full df
	df_full = df_full.append(vid_df_full, ignore_index=True)
	df_vis = df_vis.append(vid_df_vis, ignore_index=True)
	df_3min = df_3min.append(vid_df_3min, ignore_index=True)
	videos_processed += 1
	total_poses += pose_set
	total_3min_poses += three_min_pose_set


cv2.destroyAllWindows()
df_full.to_pickle(path+'full6.pkl')
df_vis.to_pickle(path+'vis6.pkl')
df_3min.to_pickle(path+'3min4.pkl')
video_files_df = pd.DataFrame(video_files)
video_files_df.to_pickle(path+'video_files4.pkl')
print(f'frames_processed: {frames_processed}')


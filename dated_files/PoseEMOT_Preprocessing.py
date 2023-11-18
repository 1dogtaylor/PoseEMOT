import numpy as np
import cv2
import mediapipe as mp
import tqdm
import os
import keras
from keras.utils import pad_sequences

mp_pose = mp.solutions.pose

def calculate_distance(pt1, pt2):
    return np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)

def calculate_angle(pt1, pt2, pt3):
    v1 = [pt2[0] - pt1[0], pt2[1] - pt1[1]]
    v2 = [pt3[0] - pt2[0], pt3[1] - pt2[1]]
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def calculate_velocity(pts, prev_pts, frame_rate):

    velocity = calculate_distance(pts,prev_pts) * frame_rate
    return np.mean(velocity)

def extract_features(landmarks, prev_landmarks, frame_rate):
    distances = []
    angles = []
    velocity = []

   
    
    # Calculate distances
    distances.append(calculate_distance(landmarks[0], landmarks[16])) # Nose to right wrist
    distances.append(calculate_distance(landmarks[0], landmarks[15])) # Nose to left wrist
    distances.append(calculate_distance(landmarks[16], landmarks[24])) # Right wrist to right hip
    distances.append(calculate_distance(landmarks[15], landmarks[23])) # Left wrist to left hip
    distances.append(calculate_distance(landmarks[13], landmarks[23])) # Left elbow to left hip
    distances.append(calculate_distance(landmarks[14], landmarks[24])) # Right elbow to right hip
    
    # Calculate angles
    angles.append(calculate_angle(landmarks[16], landmarks[14], landmarks[12])) # Right wrist to right elbow to right hip
    angles.append(calculate_angle(landmarks[11], landmarks[13], landmarks[15])) # Left wrist to left elbow to left hip
    angles.append(calculate_angle(landmarks[14], landmarks[12], landmarks[24])) # Right elbow to right shoulder to right hip
    angles.append(calculate_angle(landmarks[13], landmarks[11], landmarks[23])) # Left elbow to left shoulder to left hip
    
    # Calculate velocity
    velocity.append(calculate_velocity(landmarks[0], prev_landmarks[0], frame_rate)) # Nose
    velocity.append(calculate_velocity(landmarks[11], prev_landmarks[11], frame_rate)) # Left left shoulder
    velocity.append(calculate_velocity(landmarks[12], prev_landmarks[12], frame_rate)) # Right shoulder
    velocity.append(calculate_velocity(landmarks[13], prev_landmarks[13], frame_rate)) # Left elbow
    velocity.append(calculate_velocity(landmarks[14], prev_landmarks[14], frame_rate)) # Right elbow
    velocity.append(calculate_velocity(landmarks[15], prev_landmarks[15], frame_rate)) # Left wrist
    velocity.append(calculate_velocity(landmarks[16], prev_landmarks[16], frame_rate)) # Right wrist
    velocity.append(calculate_velocity(landmarks[23], prev_landmarks[23], frame_rate)) # Left hip
    velocity.append(calculate_velocity(landmarks[24], prev_landmarks[24], frame_rate)) # Right hip
    return np.concatenate((distances, angles, velocity))

def video_to_features(video_path):
    vibobj = cv2.VideoCapture(video_path)
    success = True

    frame_rate = vibobj.get(cv2.CAP_PROP_FPS)
    features = []
    prev_landmarks = np.zeros((33, 3))
    pbar = tqdm.tqdm(total=int(vibobj.get(cv2.CAP_PROP_FRAME_COUNT))-3)
    while success:
        pbar.update(1)
        # Read a new frame
        success, frame = vibobj.read()
        if success == False:
            continue

        # Estimate human poses using a media pipe mesh predictor
        # (Assuming the media pipe mesh predictor is implemented and accessible)
        with mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:

            results = pose.process(frame)

            pose_landmarks = results.pose_landmarks
            if pose_landmarks is not None: # ensures pose is detected
                landmarks = [[lmk.x, lmk.y, lmk.z] for lmk in pose_landmarks.landmark]
                # Extract features from frame and landmarks
                extracted_features = extract_features(landmarks, prev_landmarks, frame_rate)
                prev_landmarks = landmarks
                features.append(extracted_features)
    return features

# extract feature from videos
path = "/Users/taylorbrandl/Taylor/Python/Nimbus/DroneFollower/Pose Estimation/GEMEP_Coreset_Full Body"
labels = []
all_features = []
for filename in os.listdir(path):
    if filename.endswith(".avi"):
        print(filename)
        features = video_to_features(os.path.join(path, filename))
        features = np.array(features)
        print(features.shape)
        all_features.append(features)
        # Extract the emotion from the filename and append it to the labels list
        emotion = filename.split('_')[0][1:]
        emotion = emotion[2:]
        labels.append(emotion)
    else:
        continue


max_length = max([video.shape[0] for video in all_features])
padded_features = pad_sequences(all_features, maxlen=max_length, dtype='float32', padding='post', truncating='post', value=0.0)


np.save("PoseLabelsGEMEP_FULL",labels)
np.save("PoseEncodedGEMEP_FULL",padded_features)

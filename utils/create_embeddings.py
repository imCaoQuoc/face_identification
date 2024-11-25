import os
import cv2
import math
import json
import sys
import torch
import numpy as np
import insightface
import torch.nn as nn
from tqdm import tqdm
from insightface.app import FaceAnalysis

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(os.path.join(base_dir, "model"), "wf4m_mbf_rgb.onnx")

model_pack_name = 'buffalo_l'
provider = ['CUDAExecutionProvider']

# Detect face model
detector = FaceAnalysis(name=model_pack_name, provider=provider)
detector.prepare(ctx_id=0, det_size=(640, 640))

# Extract embedding model
handler = insightface.model_zoo.get_model(model_path, provider=provider)
handler.prepare(ctx_id=0)

def extract_frames(video_path=None):
    # Path to video
    video_path = video_path

    # Array to save frame and embedding
    frames = []

    # Read video if path exists, else open camera
    if video_path is not None:
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Cannot load video.")
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break

            frames.append(frame)
            
            if cv2.waitKey(15) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    return frames

# Function extract embeddings
def detect_extract(frame):
    pose = None
    face = detector.get(frame)
    for i in face:
        pose = i.pose
    # If face is empty, return none
    if len(face) == 0:
        return None, None
    embedding = handler.get(frame, face[0])

    return embedding, pose

# Function to calculate cosine similarity between two embeddings
def cosine_similarity(embedding1, embedding2):
    embedding1_tensor = torch.tensor(embedding1)
    embedding2_tensor = torch.tensor(embedding2)
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    similarity = cos(embedding1_tensor, embedding2_tensor)
    return similarity.item()

def is_frontal(pose):
    return 1.0 <= abs(pose[0]) <= 40.0 and 0.0 <= abs(pose[1]) <= 9.0

def is_left(pose):
    return (1.0 <= abs(pose[0]) <= 6.0 or 20.0 <= abs(pose[0]) <= 40.0) and -29.0 <= pose[1] <= -20.0

def is_right(pose):
    return (1.0 <= abs(pose[0]) <= 6.0 or 20.0 <= abs(pose[0]) <= 40.0) and 25.0 <= pose[1] <= 35.0

def get_best_embeddings(video_path=None):
    # Init dictionaries and lists to store embeddings
    storage_embeddings = []
    poses = []
    frontal = []
    lefts = []
    rights = []
    
    # Extract embedding from frame & append to array
    frames = extract_frames(video_path)
    for frame in tqdm(frames):
        embedding, pose = detect_extract(frame)
        if embedding is not None:
            storage_embeddings.append(embedding)
            poses.append(pose)

    # Convert embeddings to save to JSON
    storage_embeddings_as_lists = [embedding.tolist() for embedding in storage_embeddings]

    # Separate embeddings into frontal, left, and right groups
    for i in tqdm(range(len(poses))):
        pose = poses[i]
        embedding = storage_embeddings[i]
        
        if is_frontal(pose):
            frontal.append(embedding)
        if is_left(pose):
            lefts.append(embedding)
        if is_right(pose):
            rights.append(embedding)

    # Calculate the centers (mean) of each group
    center_frontal = np.mean(frontal, axis=0) if frontal else None
    center_left = np.mean(lefts, axis=0) if lefts else None
    center_right = np.mean(rights, axis=0) if rights else None

    # Select best embedding based on cosine similarity to center
    best_front = None
    best_left = None
    best_right = None
    max_similarity = -1  # Initialize with -1 for comparison

    # Finding the best frontal embedding
    if center_frontal is not None:
        for front in frontal:
            similarity = cosine_similarity(front, center_frontal) 
            if similarity > max_similarity:
                max_similarity = similarity
                best_front = front

    # Finding the best left embedding
    max_similarity = -1  # Reset max similarity for left
    if center_left is not None:
        for left in lefts:
            similarity = cosine_similarity(left, center_left) 
            if similarity > max_similarity:
                max_similarity = similarity
                best_left = left

    # Finding the best right embedding
    max_similarity = -1  # Reset max similarity for right
    if center_right is not None:
        for right in rights:
            similarity = cosine_similarity(right, center_right) 
            if similarity > max_similarity:
                max_similarity = similarity
                best_right = right

    return best_front, best_left, best_right
        
def create_user(video_path=None, user_name=None):
    file_path = os.path.join(base_dir, "users_data.json")

    # Checking if users_data.json existed
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
    else:
        data = []
    if user_name is None:
        return {"error": "Please enter a valid user name."}
    # Checking if user_name existed
    for user in data:
        if user["name"] == user_name:
            return {"error": "User name already exists. Please choose another name."}

    # Creating new user if user_name does not exist
    embeddings = get_best_embeddings(video_path)
    embeddings = [embedding.tolist() for embedding in embeddings]
    new_user = {
        "name": user_name,
        "embeddings": embeddings
    }
    data.append(new_user)

    # Save data into JSON file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

    return {"message": "User added successfully!"}
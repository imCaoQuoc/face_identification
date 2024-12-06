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
import networkx as nx
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(os.path.join(base_dir, "model"), "wf4m_r18_rgb.onnx")

model_pack_name = 'buffalo_l'
provider = ['CUDAExecutionProvider']

# Detect face model
detector = FaceAnalysis(name=model_pack_name, provider=provider)
detector.prepare(ctx_id=0, det_size=(640, 640))

# Extract embedding model
handler = insightface.model_zoo.get_model(model_path, provider=provider)
handler.prepare(ctx_id=0)

# def extract_frames(video_path=None):
#     # Path to video
#     video_path = video_path

#     # Array to save frame and embedding
#     frames = []

#     # Read video if path exists, else open camera
#     if video_path is not None:
#         cap = cv2.VideoCapture(video_path)

#     if not cap.isOpened():
#         print("Cannot load video.")
#     else:
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         while True:
#             ret, frame = cap.read()
            
#             if not ret:
#                 break

#             frames.append(frame)
            
#             if cv2.waitKey(15) & 0xFF == ord('q'):
#                 break
        
#         cap.release()
#         cv2.destroyAllWindows()

#     return frames

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
        frame_to_skip = int(fps) * 1  
        frame_count = 0
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if frame_count % frame_to_skip ==0:
                frames.append(frame)
            
            if cv2.waitKey(15) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
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
    return 0.0 <= abs(pose[0]) <= 40.0 and 0.0 <= abs(pose[1]) <= 9.0

def is_left(pose):
    return 0.0 <= abs(pose[0]) <= 40.0 and -45.0 <= pose[1] <= 0.0

def is_right(pose):
    return 0.0 <= abs(pose[0]) <= 40.0 and 0 <= pose[1] <= 45.0

def find_best_with_graph(embeddings, threshold=0.7):
    if len(embeddings) == 0:
        return None

    # Build cosine similarity matrix
    similarity_matrix = np.zeros((len(embeddings), len(embeddings)))
    for i in tqdm(range(len(embeddings)), desc="Building cosine similarity matrix..."):
        for j in range(len(embeddings)):
            if i != j:
                similarity_matrix[i][j] = cosine_similarity(embeddings[i], embeddings[j])

    # Create adjacency matrix based on threshold
    adjacency_matrix = (similarity_matrix > threshold).astype(int)

    # Build graph using NetworkX
    graph = nx.from_numpy_array(adjacency_matrix)

    # Find node with highest degree centrality
    degree_centrality = nx.degree_centrality(graph)
    best_node = max(degree_centrality, key=degree_centrality.get)

    return embeddings[best_node] 

def get_best_embeddings(video_path=None):
    # Init dictionaries and lists to store embeddings
    storage_embeddings = []
    poses = []
    frontal = []
    lefts = []
    rights = []
    
    # Extract embedding from frame & append to array
    frames = extract_frames(video_path)
    for frame in tqdm(frames, desc="Detecting and Extracting embeddings..."):
        embedding, pose = detect_extract(frame)
        if embedding is not None:
            storage_embeddings.append(embedding)
            poses.append(pose)

    # Convert embeddings to save to JSON
    storage_embeddings_as_lists = [embedding.tolist() for embedding in storage_embeddings]

    # Separate embeddings into frontal, left, and right groups
    for i in tqdm(range(len(poses)), desc="Choosing embeddings..."):
        pose = poses[i]
        embedding = storage_embeddings[i]
        
        if is_frontal(pose):
            frontal.append(embedding)
        if is_left(pose):
            lefts.append(embedding)
        if is_right(pose):
            rights.append(embedding)

    # Use graph-based method to find the best embedding in each group
    best_front = find_best_with_graph(frontal, threshold=0.7)
    best_left = find_best_with_graph(lefts, threshold=0.7)
    best_right = find_best_with_graph(rights, threshold=0.7)

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
    print(embeddings)
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
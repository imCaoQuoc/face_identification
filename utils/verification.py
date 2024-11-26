import json
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import insightface
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

# Function extract embeddings
def detect_extract(frame):
    face = detector.get(frame)
    # If face is empty, return none
    if len(face) == 0:
        return None
    embedding = handler.get(frame, face[0])

    return embedding

# Function to capture image from camera and extract embedding
def read_img_and_get_embedding(img=None):
    img = cv2.imread(img)
    embeddings = detect_extract(img)

    return embeddings

# Function to calculate cosine similarity between two embeddings
def cosine_similarity(embedding1, embedding2):
    embedding1_tensor = torch.tensor(embedding1)
    embedding2_tensor = torch.tensor(embedding2)
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    similarity = cos(embedding1_tensor, embedding2_tensor)
    return similarity.item()

# Function to load user data from JSON file
def load_user_data(file_path="users_data.json"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Function to verify user by comparing input embedding with stored embeddings
def verify_user(input_embedding, threshold=0.60):
    data = load_user_data()
    
    for user in data:
        name = user["name"]
        user_embeddings = [np.array(embedding) for embedding in user["embeddings"]]
        
        for user_embedding in user_embeddings:
            similarity = cosine_similarity(input_embedding, user_embedding)
            if similarity >= threshold:
                print(f"User verified as {name} with similarity: {similarity:.2f}")
                return name, similarity
    
    print("User verification failed.")
    return None, None

def run_verification(img=None):
    print("Starting verification...")
    input_embedding = read_img_and_get_embedding(img)
    if input_embedding is None:
        print("Verification aborted.")
        return
    
    name, similarity = verify_user(input_embedding)
    return name, similarity
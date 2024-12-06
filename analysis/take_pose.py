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

# Hàm để trích xuất pose
def detect_extract(frame):
    pose = None
    faces = detector.get(frame)
    for i in faces:
        pose = i.pose
    if len(faces) == 0:
        return None
    return pose

# Hàm trích xuất frames từ video
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
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_to_skip = fps * 1
        frame_count = 0
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            if frame_count % frame_to_skip == 0:
                frames.append(frame)
            frame_count += 1
            
            if cv2.waitKey(15) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    return frames

# Hàm lấy pose từ frames và thêm vào file JSON
def get_best_embeddings(video_path=None, output_file='poses.json'):
    poses = []

    # Kiểm tra nếu file đã tồn tại và đọc dữ liệu cũ
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            poses = json.load(f)

    frames = extract_frames(video_path)
    for frame in tqdm(frames):
        pose = detect_extract(frame)
        if pose is not None:
            # Chuyển đổi pose thành list để lưu vào JSON
            poses.append(pose.tolist())

    # Lưu tất cả poses vào file JSON
    with open(output_file, 'w') as f:
        json.dump(poses, f, indent=4)
    print(f"Đã thêm {len(frames)} pose vào file {output_file}")

# Sử dụng hàm với video đầu vào
get_best_embeddings("/home/quocnc1/Documents/arcface_test/toanbd1fw152_240405_session2.mp4")
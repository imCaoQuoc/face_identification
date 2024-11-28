from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse 
from fastapi.staticfiles import StaticFiles 
from utils.create_embeddings import create_user 
from utils.verification import run_verification 
from pathlib import Path
import shutil
import uvicorn 
import cv2
import os
import sys 

app = FastAPI()

# def create_tmp_video() -> str:
#     output_path = "/home/quocnc1/Documents/enroll_flow/project/download_video/output.mp4"
#     cap = cv2.VideoCapture(0)
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         out.write(frame)
#         cv2.imshow('frame', frame)
#         c = cv2.waitKey(1)
#         if c & 0xFF == ord('q'):
#             break
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

#     return output_path

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static") 

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open(os.path.join(static_dir, "index.html"), "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.api_route("/create_user_camera/", methods=["POST"])
async def create_user_endpoint(name: str = Form(...), video: UploadFile = File(...)):
    video_path = f"temp_{name}.mp4"
    with open(video_path, "wb") as buffer:
        buffer.write(await video.read())
    
    result = create_user(video_path=video_path, user_name=name)
    os.remove(video_path)
    
    if "error" in result:
        return {"error": result["error"]}
    return {"message": result["message"]}

@app.api_route("/create_user/", methods=["POST"])
async def create_user_endpoint(name: str = Form(...), video: UploadFile = File(...)):
    video_path = f"temp_{name}.mp4"
    with open(video_path, "wb") as buffer:
        buffer.write(await video.read())
    
    result = create_user(video_path, user_name=name)
    os.remove(video_path)

    if "error" in result:
        return {"error": result["error"]}
    return {"message": result["message"]}

@app.api_route("/verify_user/", methods=["POST"])
async def verify_user_endpoint(image: UploadFile = File(...)):
    image_path = "temp_image.jpg"
    with open(image_path, "wb") as buffer:
        buffer.write(await image.read())
    
    name, similarity = run_verification(image_path)

    os.remove(image_path)
    if name == None:
        return {"error": "User verification failed, please try again."}
    return {"message": f"User verified as {name} with similarity: {similarity:.2f}"}
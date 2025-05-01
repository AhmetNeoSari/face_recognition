from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dataclasses import dataclass, field
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from fastapi import WebSocket,  UploadFile, File, Form, HTTPException
import cv2
from multiprocessing import Process, Queue, Value, Manager
from starlette.responses import StreamingResponse
import time
import os
from typing import Any
import shutil

@dataclass
class WebApp:
    video_queue: Queue
    log_queue: Queue
    play_flag: Value
    shared_video_data: Manager  # Variable for video path
    base_dir : Path
    logger : Any
    app: FastAPI = field(default_factory=FastAPI, init=False)

    def __post_init__(self):
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.logger.debug("web server begin")
        self.setup_routes()
        self.mount_static_files()
        
    def setup_routes(self):
        # Home page route
        @self.app.get("/", response_class=HTMLResponse)
        def read_root():
            html_content = Path("app/web_interface/templates/index.html").read_text()
            return HTMLResponse(content=html_content)
        
        @self.app.get("/user_register", response_class=HTMLResponse)
        def read_root():
            html_content = Path("app/web_interface/templates/user_register.html").read_text()
            return HTMLResponse(content=html_content)

        @self.app.post("/set-line")
        async def set_counting_line(line: dict):
            self.shared_video_data['line_start'] = (line["start"]["x"], line["start"]["y"])
            self.shared_video_data['line_end'] = (line["end"]["x"], line["end"]["y"])
            self.logger.debug(f"Line coordinates updated: {self.shared_video_data['line_start']} to {self.shared_video_data['line_end']}")
            return JSONResponse(content={
                "message": "Line coordinates updated",
                "line": self.shared_video_data['line_start'] + self.shared_video_data['line_end']
            })

        # For video streaming
        @self.app.get("/video-stream")
        def video_stream():
            return StreamingResponse(self.video_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

        # Sending a message to the log screen
        @self.app.get("/log-stream")
        def log_stream():
            return StreamingResponse(self.log_generator(), media_type="text/event-stream")

        # Endpoint to change the playback flag
        @self.app.post("/start-video")
        def start_video():
            self.play_flag.value = 1
            return {"message": "Video başlatıldı"}

        @self.app.post("/upload-video")
        def upload_video(video_path: str = Form(...)):
            if os.path.exists(video_path):
                self.shared_video_data['video_path'] = video_path  # We set the video path
                self.play_flag.value = 0  # Stop because the video has been uploaded
                print(f"Video path set: {self.shared_video_data['video_path']}")
                return {"message": f"Video path {video_path} selected", "path": video_path}
            else:
                return {"message": "No valid video file found", "error": True}
            
        # RTSP endpoint entering endpoint
        @self.app.post("/upload-rtsp")
        def upload_rtsp(rtsp_endpoint: str = Form(...)):
            if rtsp_endpoint.startswith("rtsp://"):
                self.shared_video_data['video_path'] = rtsp_endpoint  # Setting the RTSP endpoint as a path
                self.play_flag.value = 0  # Playback is stopped, waiting for RTSP
                print(f"RTSP endpoint set: {self.shared_video_data['video_path']}")
                return {"message": f"RTSP endpoint {rtsp_endpoint} selected", "path": rtsp_endpoint}
            else:
                return {"message": "No valid RTSP endpoint found", "error": True}
            
        # Camera selection endpoint
        @self.app.post("/upload-camera")
        def upload_camera(camera_index: int = Form(...)):
            try:
                camera_index = int(camera_index)
                self.shared_video_data['video_path'] = str(camera_index)  # Setting camera index as path
                self.play_flag.value = 0  # Playback pauses, camera waiting
                print(f"Camera index set: {self.shared_video_data['video_path']}")
                return {"message": f"Camera {camera_index} selected", "camera_index": camera_index}
            except ValueError:
                return {"message": "No valid camera index found", "error": True}

        @self.app.post("/submit")
        def create_upload(
            name: str = Form(...),
            surname: str = Form(...),
            photos: list[UploadFile] = File(...)
        ):
            try:
                # Create directory name
                dir_name = f"{name.strip().lower()}_{surname.strip().lower()}"
                user_dir = self.base_dir / dir_name
                
                # Create directory
                os.makedirs(user_dir, exist_ok=True)
                
                # Save photos
                saved_files = []
                for photo in photos:
                    file_path = user_dir / photo.filename
                    
                    # Save file as synchronous
                    with open(file_path, "wb") as buffer:
                        shutil.copyfileobj(photo.file, buffer)
                    
                    saved_files.append(str(file_path))

                return {
                    "status": "success",
                    "user": f"{name} {surname}",
                    "saved_files": saved_files,
                    "directory": str(user_dir)
                }
                
            except Exception as e:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Dosya kaydetme hatası: {str(e)}"
                )
            finally:
                # Dosya pointer'ları kapat
                for photo in photos:
                    photo.file.close()
                    

    def video_generator(self):
        """ Receives frames from the queue and sends them with StreamingResponse """
        while True:
            frame = self.video_queue.get()
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    def log_generator(self):
        """Send log messages from queue to client"""
        while True:
            try:
                while not self.log_queue.empty():
                    log_message = self.log_queue.get_nowait()
                    yield f"data: {log_message}\n\n"
                time.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Log generator error: {str(e)}")
                time.sleep(0.5)

    def mount_static_files(self):
        # Mount operation for static files
        self.app.mount("/static", StaticFiles(directory="app/web_interface/static"), name="static")


# Face Recognition System

This project is a face recognition system that performs tasks such as face detection, face recognition, and face tracking. The project uses various algorithms and models to detect, recognize, and track faces.

## Project Structure

The project has the following directory structure:

FACE_RECOGNITION/
├── App/
│ ├── configs/
│ │ ├── config.py
│ │ └── config.toml
│ ├── face_detection/
│ │ ├── scrfd/
│ │ | ├── face_detector.py
│ │ | ├── weights/
│ │ | | └── README.md
│ ├── face_recognition/
│ │ ├── arcface/
│ │ | ├── recognize.py
│ │ | ├── recognizer_utils.py
│ │ | ├── update_database.py
│ │ | ├── datasets/
│ │ │ | ├── backup/
│ │ │ | ├── data/
│ │ │ | ├── face_features/
│ │ │ | └── new_persons/
│ │ │ ├── weights/
│ │ | | └── README.md
│ ├── face_tracking/
│ │ ├── byte_tracker.py
│ │ └── tracker_utils.py
├── .gitignore
├── app.py
├── README.md
└── requirements.txt




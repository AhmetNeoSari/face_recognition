# Face Recognition System

This project is a face recognition system that performs tasks such as face detection, face recognition, and face tracking. The project uses various algorithms and models to detect, recognize, and track faces.

<p align="center">
<img src="./assets/face_recognition.gif" alt="Face Recognition" />
<br>
<em>Face Recognition</em>
</p>


## Table of Contents

- [Project Structure](#Project-Structure)
- [How to use](#how-to-use)
  - [Create Environment and Install Packages](#create-environment-and-install-packages)
  - [Add new persons to datasets](#add-new-persons-to-datasets)
  - [Delete person from datasets](#Delete-person-from-datasets)
  - [View contacts saved in the datasets](#View-contacts-saved-in-the-datasets)
- [Technology](#technology)
  - [Face Detection](#face-detection)
  - [Face Recognition](#face-recognition)
  - [Face Tracking](#face-tracking)

## Project Structure

The project has the following directory structure:

```
.
├── app
│   ├── config.py
│   ├── face_detection
│   │   ├── __init__.py
│   │   └── scrfd
│   │       ├── face_detector.py
│   │       └── weights
│   │           ├── README.md
│   │           └── scrfd_2.5g_bnkps.onnx
│   ├── face_recognition
│   │   ├── arcface
│   │   │   ├── datasets
│   │   │   │   ├── backup
│   │   │   │   │   └── ahmet_sari
│   │   │   │   │       ├── 2024-08-14-131926.jpg
│   │   │   │   │       ├── 2024-08-14-131927.jpg
│   │   │   │   │       ├── 2024-08-14-131928.jpg
│   │   │   │   │       ├── 2024-08-14-131929.jpg
│   │   │   │   │       └── 2024-08-14-131931.jpg
│   │   │   │   ├── data
│   │   │   │   │   └── ahmet_sari
│   │   │   │   │       ├── 0.jpg
│   │   │   │   │       ├── 1.jpg
│   │   │   │   │       ├── 2.jpg
│   │   │   │   │       ├── 3.jpg
│   │   │   │   │       └── 4.jpg
│   │   │   │   ├── face_features
│   │   │   │   │   └── feature.npz
│   │   │   │   └── new_persons
│   │   │   ├── __init__.py
│   │   │   ├── recognize.py
│   │   │   ├── recognizer_utils.py
│   │   │   ├── update_database.py
│   │   │   └── weights
│   │   │       ├── arcface_r100.pth
│   │   │       └── README.md
│   │   └── __init__.py
│   ├── face_tracking
│   │   ├── byte_tracker.py
│   │   ├── __init__.py
│   │   └── tracker_utils.py
│   ├── fps.py
│   ├── __init__.py
│   ├── logger.py
│   ├── streamer.py
│   └── utils.py
├── app.py
├── assets
│   ├── add_person.png
│   ├── bytetrack.png
│   ├── config_local_toml.png
│   ├── delete_person.png
│   ├── face_recognition.gif
│   └── list_people.png
├── configs
│   ├── config.local.toml
│   └── config.prod.toml
├── logs
│   └── app.log
├── README.md
└── requirements.txt

```

## How to use

### Create Environment and Install Packages
check if you have cuda installed on your computer
```bash
nvcc --version
```

If cuda is not installed on your computer
- [NVIDIA CUDA INTALLATION GUIDE](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

Check if Conda environment is installed
```bash
conda --version
```
if conda environment is not installed please do the necessary installations

- [Conda setup](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html)

```bash
conda create -n face-dev python=3.9
```

```bash
conda activate face-dev
```

Please install the Torch library compatible with cuda on your system. In my case the following command works. Please do the necessary research from the related link (choose one of the them)
- [PyTorch Installation Guıde](https://pytorch.org/get-started/locally/)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
if you don't have CUDA
```bash
pip install torch==1.9.1+cpu torchvision==0.10.1+cpu torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

if you don't have gpu use "onnxruntime==1.18.1" instead "onnxruntime-gpu==1.18.1" in requirements.txt

```bash
pip install -r requirements.txt
```
## Install weights for detection and recognition
to install detection model weight go to weights directory
```bash
cd app/face_recognition/arcface/weights/
```
and from the url in readme.md install scrfd_2.5g_bnkps.onnx file
- [link](https://drive.google.com/drive/folders/1C9RzReAihJQRl8EJOX6vQj7qbHBPmzME?usp=sharing)

to install recognition model weight go to weights directory
```bash
cd app/face_recognition/arcface/weights/ 
```

Download arcface_r100.pth at the link
- [link](https://drive.google.com/drive/folders/1CHHb_7wbvfjKPFNKVBb76lL5sVfBLcv5?usp=sharing)

To run the project you need to load data into the Dataset. For this you need to add at least one person to dataset
```bash
cd ../../..
```
Please make sure you are in the face_recognition(root) directory!

### Config Files

The config.local.toml files contain configuration information such as model paths and 
detection/recognition/tracking parameters. Edit these files according to your needs:
(You can create similar configuration for production with necessary paths and parameters)

### Running the Application
For local environment:

```bash
python app.py --env local
```
For production environment:
```bash
python app.py --env prod
```

## Technology

### Face Detection

 **SCRFD**
   - SCRFD (Single-Shot Scale-Aware Face Detector) is designed for real-time face detection across various scales. It is particularly effective in detecting faces at different resolutions within the same image.

### Face Recognition

 **ArcFace**

   - ArcFace is a state-of-the-art face recognition algorithm that focuses on learning highly discriminative features for face verification and identification. It is known for its robustness to variations in lighting, pose, and facial expressions.


### Face Tracking

 **ByteTrack**
   <p align="center">
   <img src="./assets/bytetrack.png" alt="ByteTrack" />
   <br>
   <em>ByteTrack is a simple, fast and strong multi-object tracker.</em>
   </p>

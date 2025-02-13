import os
import shutil
import warnings 
import sys

import cv2
import numpy as np
import torch
from torchvision import transforms
from dataclasses import dataclass
from typing import Any, List, Tuple
from collections import Counter


@dataclass
class UpdateDatabase:

    """
    This class adds new people's information to the program. 
    Allows the system to recognize the person

    This class allows the model to learn by extracting the features of the faces of
    new people added to the program. The images of the learned people are preserved
    in the database as a recovery.
    """

    backup_dir      : str  
    add_persons_dir : str 
    faces_save_dir  : str 
    features_path   : str
    recognizer_model_name : str
    recognizer_model_path : str
    logger : Any
    
    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the face recognizer
        self.recognizer = iresnet_inference(
            model_name=self.recognizer_model_name, 
            path=self.recognizer_model_path, 
            device=self.device
        )
        self.logger.info('Update database application started')


    @torch.no_grad()
    def get_feature(self, face_image):
        """
        Extract facial features from an image using the face recognition model.

        Args:
            face_image (numpy.ndarray): Input facial image.

        Returns:
            numpy.ndarray: Extracted facial features.
        """
        # Define a series of image preprocessing steps
        face_preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((112, 112)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # Convert the image to RGB format
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Apply the defined preprocessing to the image
        face_image = face_preprocess(face_image).unsqueeze(0).to(self.device)

        # Use the model to obtain facial features
        emb_img_face = self.recognizer(face_image)[0].cpu().numpy()

        # Normalize the features
        images_emb = emb_img_face / np.linalg.norm(emb_img_face)
        return images_emb


    def detect_and_save_faces(self, person_image_path:str, person_face_path:str):
        """
        Detect faces in the images and save them.

        Args:
            person_image_path (str): Path to the person's images.
            person_face_path  (str): Path to save detected faces.

        Returns:
            tuple: Lists of image names and image embeddings.
        """
        images_name = []
        images_emb = []

        for image_name in os.listdir(person_image_path):
            if image_name.lower().endswith(("png", "jpg", "jpeg")):
                input_image = cv2.imread(os.path.join(person_image_path, image_name))

                # Detect faces and landmarks using the face detector
                try:
                    bboxes, landmarks = self.detector.detect(image=input_image)
                except Exception as e:
                    self.logger.error(f"Error when detect {e}")
                    sys.exit(1)

                for i, (x1, y1, x2, y2, _) in enumerate(bboxes):
                    # Get the number of files in the person's path
                    number_files = len(os.listdir(person_face_path))

                    # Extract the face from the image
                    face_image = input_image[int(y1):int(y2), int(x1):int(x2)]

                    # Path to save the face
                    path_save_face = os.path.join(person_face_path, f"{number_files}.jpg")

                    # Save the face to the database
                    cv2.imwrite(path_save_face, face_image)

                    # Extract features from the face
                    images_emb.append(self.get_feature(face_image=face_image))
                    images_name.append(os.path.basename(person_image_path))

        return np.array(images_name), np.array(images_emb)


    def update_database(self, images_name: np.ndarray, images_emb: np.ndarray):
        """
        Update the face features database.

        Args:
            images_name   (numpy.ndarray) : List of image names.
            images_emb    (numpy.ndarray) : List of image embeddings.
        """

        # Read existing features if available
        features = self.read_features()
        if features is not None:
            # Unpack existing features
            old_images_name, old_images_emb = features
            # Combine new features with existing features
            if len(old_images_name) != 0 and len(old_images_emb) != 0 :
                images_name = np.hstack((old_images_name, images_name))
                images_emb  = np.vstack((old_images_emb, images_emb))
            
            self.logger.info('Update features!')

        # Save the combined features
        np.savez_compressed(self.features_path, images_name=images_name, images_emb=images_emb)


    def backup_new_persons(self):
        """
        Backup the data of the new persons.
        """
        # Move the data of the new person to the backup data directory
        for sub_dir in os.listdir(self.add_persons_dir):
            dir_to_move = os.path.join(self.add_persons_dir, sub_dir)
            shutil.move(dir_to_move, self.backup_dir, copy_function=shutil.copytree)


    def create_directories_if_not_exists(self, directories):
        for directory_path in directories:
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
                self.logger.info(f"path '{directory_path}' created.")


    def add_persons(self,detector):
        """
        Add a new person to the face recognition database.

        Args:
            detector (Face_Detector): 
        """
        # Initialize lists to store names and features of added images
        self.detector = detector
        all_images_name = np.array([], dtype=str)
        all_images_emb = np.empty((0, 512))  # assuming embeddings have size 512

        self.create_directories_if_not_exists([self.backup_dir, self.add_persons_dir, self.faces_save_dir])
        skipped_folders = 0
        # Read the folder with images of the new person, extract faces, and save them
        for name_person in os.listdir(self.add_persons_dir):
            if name_person.count('_') != 1:
                self.logger.warning(f"Skipped invalid folder name: {name_person}. Folder name must contain exactly one underscore.")
                skipped_folders += 1
                continue
            person_image_path = os.path.join(self.add_persons_dir, name_person)

            # Create a directory to save the faces of the person
            person_face_path = os.path.join(self.faces_save_dir, name_person)
            if os.path.exists(person_face_path):
                self.logger.warning(f"created {person_face_path} folder already exists")
            os.makedirs(person_face_path, exist_ok=True)

            # Detect and save faces
            print("aaaa")
            images_name, images_emb = self.detect_and_save_faces(person_image_path, person_face_path)
            print("bbbb")
            all_images_name = np.concatenate((all_images_name, images_name))
            all_images_emb = np.vstack((all_images_emb, images_emb))

        # Check if no new person is found
        if all_images_name.size == 0 and all_images_emb.size == 0 :
            self.logger.error("No new person found")
            return None
        
        # Update the database with the new features
        self.update_database(all_images_name, all_images_emb)

        # Backup the new persons data
        self.backup_new_persons()
        self.logger.info("Successfully added new person!")



    def fetch_images(self, source_path: str):
        """
        Moves only images from subfolders in the source path to self.add_persons_dir.

        Creates a folder for each subfolder in self.add_persons_dir and
        moves only image files into these folders.
        
        Args:
            source_path (str): Path to the source folder (e.g., "webcam").
        """
        self.create_directories_if_not_exists([self.add_persons_dir])
        
        total_folders = 0
        total_images = 0
        skipped_folders = 0
        for item in os.listdir(source_path): # item -> ahmet_sari
            item_path = os.path.join(source_path, item) # item_path -> home/ahmet/webcam/pictures/ahmet_sari
            if os.path.isdir(item_path):
                if item.count('_') != 1:
                    self.logger.warning(f"Skipped invalid folder name: {item}. Folder name must contain exactly one underscore.")
                    skipped_folders += 1
                    continue

                target_path = os.path.join(self.add_persons_dir, item) # target_path -> new_person/ahmet_sari
                if os.path.exists(target_path):
                    self.logger.warning(f"the photos of the person you are trying to add are already in the add folder {target_path}")
                os.makedirs(target_path, exist_ok=True)
                
                image_count = 0
                for file in os.listdir(item_path):
                    if file.lower().endswith(("png", "jpg", "jpeg")):
                        source_file = os.path.join(item_path, file)
                        target_file = os.path.join(target_path, file)
                        
                        # If there is a file with the same name, create a new name
                        base, extension = os.path.splitext(file)
                        i = 1
                        while os.path.exists(target_file):
                            target_file = os.path.join(target_path, f"{base}_{i}{extension}")
                            i += 1
                        
                        shutil.move(source_file, target_file)
                        image_count += 1
                        self.logger.debug(f"Copied: {source_file} -> {target_file}")
                os.rmdir(item_path)
                if image_count > 0:
                    total_folders += 1
                    total_images += image_count
                    self.logger.debug(f"Created folder and moved images: {item_path} -> {target_path}")
                    self.logger.debug(f"Images in {item}: {image_count}")
                else:
                    os.rmdir(target_path)  # If there is no image, delete the created folder

        self.logger.info(f"{total_images} images from {total_folders} folders copied to {self.add_persons_dir}.")


    def delete_persons(self, persons_list: List[Tuple[str, str]]):
        """
        Delete multiple persons' data from the system.

        This function deletes data for multiple persons registered in the system. For each person,
        it removes their face photos folder, backup folder, and facial features from the extracted features file.

        Args:
            persons_list (List[Tuple[str, str]]): List of tuples containing (first_name, last_name) of persons to delete.
        """
        try:
            for person_first_name, person_last_name in persons_list:
                try:
                    if person_last_name.strip() == "":
                        self.logger.error(f"Last name is required for {person_first_name} !!!")
                        continue
                    self._delete_single_person(person_first_name, person_last_name)
                except Exception as e:
                    self.logger.error(f"Error deleting {person_first_name} {person_last_name}: {e}")
        except Exception as e:
            print("person_last_name",person_last_name)
            if not person_last_name or person_last_name.strip() == "":
                self.logger.error(f"Last name is required for {person_first_name}")

            self.logger.error(f"Error occured when iterate persons {e}")
            
    def _delete_single_person(self, person_first_name: str, person_last_name: str):
        """
        Helper function to delete a single person's data.
        """
        person_name = f"{person_first_name}_{person_last_name}".lower()
        person_face_path = os.path.join(self.faces_save_dir, person_name)
        person_backup_path = os.path.join(self.backup_dir, person_name)

        # Remove the person's directories
        for path in [person_face_path, person_backup_path]:
            if os.path.exists(path):
                shutil.rmtree(path, ignore_errors=False)
                self.logger.debug(f"Deleted directory: {path}")
            else:
                self.logger.warning(f"Directory not found: {path}")

        # Load existing features
        features = self.read_features()
        if features is None:
            self.logger.error("No Features data found")
            return 

        images_name, images_emb = features
        indices_to_remove = [i for i, name in enumerate(images_name) if name == person_name]

        if len(indices_to_remove) == 0:
            self.logger.warning(f"Person {person_first_name} {person_last_name} not found in features")
            return

        # Remove the person's features from the database
        images_name = np.delete(images_name, indices_to_remove)
        images_emb = np.delete(images_emb, indices_to_remove, axis=0)

        # Save the updated features
        np.savez_compressed(self.features_path, images_name=np.array(images_name), images_emb=np.array(images_emb))
        self.logger.info(f"Person {person_first_name} {person_last_name} deleted from features")

    def read_features(self):
        try:
            data = np.load(self.features_path + ".npz", allow_pickle=True)
            images_name = data["images_name"]
            images_emb = data["images_emb"]

            return images_name, images_emb
        except:
            return None
        

    def count_persons_and_photos(self):
        """
        Count the number of persons and the number of photos for each person in the data directory.
        """

        features = self.read_features()
        if features is None:
            self.logger.error("No found Features data")
            sys.exit(1)

        images_name, _ = features
        name_counts = Counter(images_name)
        num_people = len(name_counts)
        
        self.logger.info(f"Number of people registered in the system is {num_people}")

        for name, photo_number in name_counts.items():
            self.logger.info(f"{name} has: {photo_number} photos")


if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    root_dir = os.path.dirname(parent_dir)
    sys.path.append(root_dir)
    sys.path.append(current_dir)

    from face_detection.scrfd.face_detector import Face_Detector
    from face_recognition.arcface.recognizer_utils import iresnet_inference
    
    class Custom_logger:
        def error(self, message: str):
            print(message)
        
        def warning(self, message: str):
            print(message)
        
        def info(self, message: str):
            print(message)
        
        def debug(self, message: str):
            print(message)
        
        def critical(self, message: str):
            print(message)
        
        def trace(self, message: str):
            print(message)
    my_dict = {
        "backup_dir"       : "datasets/backup",
        "add_persons_dir"  : "datasets/new_persons",
        "faces_save_dir"   : "datasets/data",
        "features_path"    : "datasets/face_features/feature",
        "recognizer_model_name" : "r100",
        "recognizer_model_path" : "weights/arcface_r100.pth"
    }

    detector_dict = {
        "model_file" : "../../face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx",
        "taskname" : "detection",
        "batched" : False,
        "nms_thresh" : 0.4,
        "session" : "",
        "detect_thresh" : 0.5,
        "detect_input_size" : [128, 128],
        "max_num" : 0,
        "metric" : "default",
        "scalefactor" : 0.0078125 #1.0 / 128.0
    }


    logger = Custom_logger()
    detector = Face_Detector(**detector_dict, logger=logger)
    obj = UpdateDatabase(**my_dict, logger=logger)
    # obj.fetch_images("/home/ahmet/Pictures/persons") #TODO edit this line
    obj.add_persons(detector=detector)
    
    # to delete a user 
    # delete_list =[
    #                 ("arif", "erol"),
    #                 ("ahmet", "sari"),
    #                 ("enes", "ak")
    #             ]
    # obj.delete_persons(delete_list)

    #to list users
    # obj.count_persons_and_photos()

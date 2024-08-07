import os
import shutil
import warnings 
import sys

import cv2
import numpy as np
import torch
from torchvision import transforms
from dataclasses import dataclass


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.append(root_dir)
sys.path.append(current_dir)

print(current_dir)

from face_detection.scrfd.face_detector import Face_Detector
from face_recognition.arcface.rocognizer_utils import iresnet_inference

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
    debug : bool
    
    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the face recognizer
        self.recognizer = iresnet_inference(
            model_name=self.recognizer_model_name, 
            path=self.recognizer_model_path, 
            device=self.device
        )


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
                bboxes, _ = self.detector.detect(image=input_image)

                # Extract faces
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
            
            if self.debug:
                print("Update features!")
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


    def add_persons(self,detector):
        """
        Add a new person to the face recognition database.

        Args:
            backup_dir      (str): Directory to save backup data.
            add_persons_dir (str): Directory containing images of the new person. ###new_persons
            faces_save_dir  (str): Directory to save the extracted faces. ###"./datasets/data/"
            features_path   (str): Path to save face features.
        """
        # Initialize lists to store names and features of added images
        self.detector = detector
        all_images_name = np.array([], dtype=str)
        all_images_emb = np.empty((0, 512))  # assuming embeddings have size 512

        # Read the folder with images of the new person, extract faces, and save them
        for name_person in os.listdir(self.add_persons_dir):
            person_image_path = os.path.join(self.add_persons_dir, name_person) #new_persons/ahmet_sari

            # Create a directory to save the faces of the person
            person_face_path = os.path.join(self.faces_save_dir, name_person) #data/ahmet_sari
            os.makedirs(person_face_path, exist_ok=True)

            # Detect and save faces
            images_name, images_emb = self.detect_and_save_faces(person_image_path, person_face_path)

            all_images_name = np.concatenate((all_images_name, images_name))
            all_images_emb = np.vstack((all_images_emb, images_emb))

        # Check if no new person is found
        if all_images_name.size == 0 and all_images_emb.size == 0 :
            if self.debug:
                print("No new person found!")
            return None
        
        # Update the database with the new features
        self.update_database(all_images_name, all_images_emb)

        # Backup the new persons data
        self.backup_new_persons()
        if self.debug:
            print("Successfully added new person!")


    def fetch_images(self, person_first_name: str, person_last_name: str, source_path: str ):
        """
        Moves photos from a specific folder to another folder.     

        Creates a folder with person's first name and last name 
        in the destination folder and transfers the images into this folder.
        
        Args:
            person_first_name (str): Name of the person.
            person_last_name  (str): Last name of the person.
            source_path       (str): Path to the source folder.
        """
        folder_name = f"{person_first_name}_{person_last_name}".lower()
        person_target_path = os.path.join(self.add_persons_dir, folder_name)

        os.makedirs(person_target_path, exist_ok=True)
        counter = 0
        for filename in os.listdir(source_path):
            if filename.lower().endswith(("png", "jpg", "jpeg")):
                source_file = os.path.join(source_path, filename)
                target_file = os.path.join(person_target_path, filename)
                shutil.move(source_file, target_file)
                counter += 1
                if self.debug:
                    print(f"Moved: {source_file} -> {target_file}")
        if self.debug:
            print(f"{counter} images moved to {person_target_path} folder.")


    def delete_person(self, person_first_name: str, person_last_name: str):
        """
        Delete a person's data from the system.

        This function deletes a person registered in the system from the folder with their face photos,
        the folder where their backups are kept. 
        Deletes facial features from the file where the person's facial features are extracted

        Args:
            person_first_name (str): First name of the person.
            person_last_name  (str): Last name of the person.
        """
        person_name = f"{person_first_name}_{person_last_name}".lower()
        person_face_path = os.path.join(self.faces_save_dir, person_name)
        person_backup_path = os.path.join(self.backup_dir, person_name)

        # Remove the person's face directory
        try: 
            for path in [person_face_path, person_backup_path]:
                if os.path.exists(path):
                    shutil.rmtree(path, ignore_errors=False)
                    print(f"Deleted directory: {path}")
                else:
                    warnings.warn(f"Directory not found: {path}")

        except Exception as e:
            warnings.warn(f"An error occurred while deleting directories: {e}")

        # Load existing features
        features = self.read_features()
        if features is None:
            warnings.warn("No found Features data")
        
        images_name, images_emb = features
        indices_to_remove = [i for i, name in enumerate(images_name) if name == person_name]

        if len(indices_to_remove) == 0:
            warnings.warn(f"Person {person_first_name} {person_last_name} doesn't found in features")

        # Remove the person's features from the database
        images_name = np.delete(images_name, indices_to_remove)
        images_emb = np.delete(images_emb, indices_to_remove, axis=0)

        # Save the updated features
        np.savez_compressed(self.features_path, images_name=np.array(images_name), images_emb=np.array(images_emb))
        if self.debug:
            print(f"Person {person_first_name} {person_last_name} deleted from database and features")


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

        Returns:
            total_persons       (int) : Total number of persons.
            person_photo_counts (dict): A dictionary where keys are person names and values are the number of photos.
        """
        person_photo_counts = {}

        # Iterate through each directory in the data directory
        for person_name in os.listdir(self.faces_save_dir):
            person_path = os.path.join(self.faces_save_dir, person_name)

            if os.path.isdir(person_path):
                # Count the number of photos for this person
                photo_count = len([f for f in os.listdir(person_path) if f.endswith(('png', 'jpg', 'jpeg'))])
                person_photo_counts[person_name] = photo_count

        total_persons = len(person_photo_counts)
        return total_persons, person_photo_counts
    

my_dict = {
    "backup_dir"       : "datasets/backup",
    "add_persons_dir"  : "datasets/new_persons",
    "faces_save_dir"   : "datasets/data",
    "features_path"    : "datasets/face_features/feature",
    "recognizer_model_name" : "r100",
    "recognizer_model_path" : "weights/arcface_r100.pth",
    "debug" : False
}

detector_dict = {
    "model_file" : "../../face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx",
    "taskname" : "detection",
    "batched" : False,
    "nms_thresh" : 0.4,
    "center_cache" : {},
    "session" : "",
    "detect_thresh" : 0.5,
    "detect_input_size" : [128, 128],
    "max_num" : 0,
    "metric" : "default",
    "scalefactor" : 0.0078125 #1.0 / 128.0
}

if __name__ == "__main__":
    detector = Face_Detector(**detector_dict)
    obj = UpdateDatabase(**my_dict)
    total_persons, person_photo_counts = obj.count_persons_and_photos()
    print(total_persons)
    print(person_photo_counts)

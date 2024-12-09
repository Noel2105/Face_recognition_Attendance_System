import os 
import face_recognition
import pickle

def generate_encodings(images_to_encode_folder_path, path_to_store_encodings):

    all_face_encodings = {}

    all_directories = os.listdir(images_to_encode_folder_path)

    for folder in all_directories:
        
        face_encodings = []

        all_files = os.listdir(f'{images_to_encode_folder_path}/{folder}')
        img_path = [filename for filename in all_files if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.gif', '.bmp'))] # Filter for image files (e.g., .jpg, .png, .jpeg)
        
        for img in img_path:
            image = face_recognition.api.load_image_file(f'{images_to_encode_folder_path}/{folder}/{img}')
            encoding = face_recognition.api.face_encodings(image, model='large')
            if len(encoding) == 1:
                face_encodings.append(encoding[0])
                print(f'Successfully generated face encodings of "{images_to_encode_folder_path}/{folder}/{img}"')
            elif len(encoding) == 0:
                print(f'No face detected in {images_to_encode_folder_path}/{folder}/{img}')
                return
            elif len(encoding) > 1:
                print(f'More than one faces detected in {images_to_encode_folder_path}/{folder}/{img}')
                return
            
        all_face_encodings[folder] = face_encodings
        
    with open(path_to_store_encodings,'wb') as file:
        pickle.dump(all_face_encodings, file)


generate_encodings(
    images_to_encode_folder_path = "./to_encode_faces",
    path_to_store_encodings = './encodings/multiple_angles_faces_encodings'
)
import os 
import face_recognition
import cv2

def validate_images(images_to_encode_folder_path):

    to_recapture = []

    all_directories = os.listdir(images_to_encode_folder_path)

    for folder in all_directories:

        all_files = os.listdir(f'{images_to_encode_folder_path}/{folder}')
        img_path = [filename for filename in all_files if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.gif', '.bmp'))] # Filter for image files (e.g., .jpg, .png, .jpeg)

        for img in img_path:
            
            image = cv2.imread(f'{images_to_encode_folder_path}/{folder}/{img}')
            
            # Detect faces in the image
            face_locations = face_recognition.face_locations(image)

            current_img_path = f'{images_to_encode_folder_path}/{folder}/{img}'
            
            if len(face_locations) == 0:
                print(f'* "{current_img_path}" - No Faces Detected! Bad!')
                to_recapture.append(current_img_path)
            elif len(face_locations) > 1:
                print(f'* "{current_img_path}" - More than one Faces Detected! Bad!')
                to_recapture.append(current_img_path)
            elif len(face_locations) == 1:
                print(f'"{current_img_path}" - Single face detected. Good.')

    if len(to_recapture) != 0:           
        print(f'\n\nThe below images either have multiple faces or cannot detect a face at all. Recapture the image of people given below to have detectable single face:')
        print(to_recapture)
    else:
        print('\n\nCan detect a single face in every image. All good. Proceed to generating encodings.')

validate_images("./to_encode_faces")
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options,
                                     num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

def print_landmarks_coordinates(hand_landmarks_list, handedness_list):
    # Dictionary of landmark names
    right_landmarks = np.zeros((2,21,3))

    
    for hand_idx, hand_landmarks in enumerate(hand_landmarks_list):
        if handedness_list[hand_idx][0].category_name == "Right":
            hand_array = [[landmark.x, landmark.y, 0] for landmark in hand_landmarks]
            landmarks_array = np.array(hand_array)
            right_landmarks[0]=landmarks_array
        if handedness_list[hand_idx][0].category_name == "Left":
            hand_array = [[landmark.x, landmark.y, 0] for landmark in hand_landmarks]
            landmarks_array = np.array(hand_array)
            right_landmarks[1]=landmarks_array

    return right_landmarks


def process(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create MediaPipe image from the RGB frame
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Detect hand landmarks
    detection_result = detector.detect(mp_image)

    # Print coordinates
    coordinates = print_landmarks_coordinates(detection_result.hand_landmarks, detection_result.handedness)

    return coordinates


def center_crop(image, target_size):
    ih, iw = image.shape[:2]
    th, tw = target_size
    img_aspect = iw / ih
    target_aspect = tw / th
    if img_aspect > target_aspect:
        new_width = int(ih * target_aspect)
        left = (iw - new_width) // 2
        image_cropped = image[:, left:left+new_width]
    else:
        new_height = int(iw / target_aspect)
        top = (ih - new_height) // 2
        image_cropped = image[top:top+new_height, :]

    image_resized = cv2.resize(image_cropped, (tw, th))
    return image_resized

# Load the trained model
model = load_model("MobileNetV3Large_50epoch_21_class.h5")

# Define class names
class_names = ["apa",
"bagus",
"berapa",
"bicara",
"bisa",
"buruk",
"iya",
"kapan",
"kasih",
"kau",
"kita",
"maaf",
"perlu",
"saya",
"sedih",
"senang",
"terima",
"tidak",
"tidak_ada",
"tolong",
"tunggu",]

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_cropped = center_crop(frame, (720, 405))  # (height, width)
    # cv2.imshow('Camera 9:16 Crop (720x1280)', frame_cropped)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    
    try:
        cords = process(frame_cropped) 
        
        # Make prediction
        if cords is not None:
            # Preprocess for model input
            if len(cords.shape) == 3 and cords.shape == (2, 21, 3):
                input_data = np.expand_dims(cords, axis=0)  
                
                # Get prediction
                predictions = model.predict(input_data, verbose=0)
                predicted_class = np.argmax(predictions, axis=1)
                confidence = np.max(predictions)
                
                prediction_text = f"{class_names[predicted_class[0]]}: {confidence:.2f}"
                cv2.putText(frame_cropped, prediction_text, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                cv2.putText(frame_cropped, "Invalid shape", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame_cropped, "No hand detected", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    except Exception as e:
        cv2.putText(frame_cropped, f"Error: {str(e)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
    cv2.imshow('Rock Paper Scissors Detector', frame_cropped)
    
    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





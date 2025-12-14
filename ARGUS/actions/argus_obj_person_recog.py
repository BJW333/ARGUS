import tensorflow as tf
import cv2
import numpy as np
import os
import mediapipe as mp
from scipy.spatial import distance
from pathlib import Path
import time
from huggingface_hub import snapshot_download

script_dir = Path(__file__).parent

#load the object recognition model
repo_id = "bjw333/ARGUS_obj_person_recog_model"  
model_path = snapshot_download(repo_id=repo_id, repo_type="model")
model = tf.saved_model.load(model_path)

#path to known faces folder
known_faces_folder = script_dir / "trainingdata/knownfaces_for_recog"

#load known faces from folder (now using landmarks)
def load_known_face_landmarks(face_mesh):
    known_faces_landmarks = {}
    for filename in os.listdir(known_faces_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(known_faces_folder, filename)
            name = os.path.splitext(filename)[0]
            img = cv2.imread(img_path)
            #detect face mesh landmarks for the known faces
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
            if results.multi_face_landmarks:
                #save the landmarks of the known face
                known_faces_landmarks[name] = results.multi_face_landmarks[0]
    return known_faces_landmarks

def compute_confidence(min_distance, min_possible_distance=100, max_possible_distance=270):
    #confidence = (max_possible_distance - min_distance) / (max_possible_distance - min_possible_distance)# old line
    confidence = max(0.0, min((max_possible_distance - min_distance) / (max_possible_distance - min_possible_distance), 1.0)) # new line here
    confidence = max(0.0, min(confidence, 1.0))  #confidence must be between 0 and 1
    return confidence

#detect faces and compare using face mesh landmarks
def detect_and_recognize_faces_with_mesh(image, face_mesh, known_faces_landmarks, confidence_threshold=0.4): #work on the confidence threshold
    recognized_faces = []
    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            best_match_name = "Unknown"
            min_distance = float('inf')

            for name, known_face_landmarks in known_faces_landmarks.items():
                #compare landmarks between detected and known face
                distances = []
                for i in range(len(face_landmarks.landmark)):
                    #get the current landmark's coordinates for both faces
                    known_x = known_face_landmarks.landmark[i].x * w
                    known_y = known_face_landmarks.landmark[i].y * h
                    detected_x = face_landmarks.landmark[i].x * w
                    detected_y = face_landmarks.landmark[i].y * h
                    #calculate distance between the landmarks
                    distances.append(distance.euclidean((known_x, known_y), (detected_x, detected_y)))

                avg_distance = np.mean(distances)
                if avg_distance < min_distance:
                    min_distance = avg_distance
                    best_match_name = name
                    
            confidence = compute_confidence(min_distance)
            
            #debugging
            #print(f"min dis: {min_distance}, confidence: {confidence}")

            if confidence < confidence_threshold:
                best_match_name = "Unknown"

            recognized_faces.append((face_landmarks, best_match_name))

    return recognized_faces


def detect_and_draw_face_mesh(image, face_mesh, face_landmarks):
    h, w, _ = image.shape
    important_indices = [
                #Outer face boundary  #done
                234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 379, 365, 397, 288, 361, 323, 454, 356, 389, 
                251, 284, 332, 297, 338, 10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 361, 389, 356, 454, 323, 361, 368,

                #Jawline
                10, 152, 234, 454, 323, 93, 132, 58, 172, 152, 
                
                #Eyebrows
                #Left Eyebrow
                70, 63, 105, 107, 52, 53, 55, 66, 64, #Left upper and lower brow

                #Right Eyebrow
                336, 285, 295, 282, 52, 53, 55, 296, 293, #Right upper and lower brow 
                
                
                #Eyes # done
                33, 133, 160, 159, 158, 144, 153, 154, # Left eye 
                263, 362, 387, 386, 385, 373, 380, 374, # Right eye

                #Nose
                #Nose Bridge
                6, 168, 195, 5, 4,
                #Left Side
                94, 129, 2, 4,
                #Right Side
                279, 358, 421, 430,
                
                #Mouth 
                78, 308, 61, 291, 13, 14, 17, 18, 80, 82, 312, 324, 37, 267, 269, 270, 273, 277, 402, 14,
                185, 191, 202, 204, 209, 211, 217, 219, 222, 227, 233, 240, 243, 246, 249, 251,
                  

            ]
    

    #landmark points
    for idx in important_indices:
        x = int(face_landmarks.landmark[idx].x * w)
        y = int(face_landmarks.landmark[idx].y * h)
        cv2.circle(image, (x, y), 5, (238, 130, 238), -1)  #draw points for landmarks

    return image


def detect_objects(image, confidence_threshold=0.5):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = model(input_tensor)

    #detections to numpy arrays
    #num_detections = int(detections.pop('num_detections')) #old line
    num_detections = int(detections.pop('num_detections', len(detections['detection_scores']))) # new line here
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}

    #confidence threshold filter
    detection_scores = detections['detection_scores']
    mask = detection_scores >= confidence_threshold

    #filter out low confidence detections
    for key in detections:
        detections[key] = detections[key][mask]

    detections['num_detections'] = len(detections['detection_scores'])
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    return detections

def objectrecognitionrun():
    #label map
    label_map = {
        1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus",
        7: "train", 8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant",
        13: "stop sign", 14: "parking meter", 15: "bench", 16: "bird", 17: "cat", 18: "dog",
        19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear", 24: "zebra",
        25: "giraffe", 27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
        34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite", 39: "baseball bat",
        40: "baseball glove", 41: "skateboard", 42: "surfboard", 43: "tennis racket", 44: "bottle",
        46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "banana",
        53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog",
        59: "pizza", 60: "donut", 61: "cake", 62: "chair", 63: "couch", 64: "potted plant",
        65: "bed", 67: "dining table", 70: "toilet", 72: "tv", 73: "laptop", 74: "mouse",
        75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave", 79: "oven", 80: "toaster",
        81: "sink", 82: "refrigerator", 84: "book", 85: "clock", 86: "vase", 87: "scissors",
        88: "teddy bear", 89: "hair drier", 90: "toothbrush"
    }

    cap = cv2.VideoCapture(0)
    start_time = time.time()
    duration = 10
    
    if not cap.isOpened():
        print("Error: Camera not accessible or not found.")
        return

    detected_labels = set()

    #initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=3) #orginal max num faces 1 

    #load known face landmarks
    known_faces_landmarks = load_known_face_landmarks(face_mesh)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        #object detection
        detections = detect_objects(frame)

        #detection classes to labels
        detection_labels = [label_map.get(class_id, 'Unknown') for class_id in detections['detection_classes']]

        for i, label in enumerate(detection_labels):
            confidence = detections['detection_scores'][i]
            detected_labels.add(label)

            #bounding box for detected objects
            bbox = detections['detection_boxes'][i]
            ymin, xmin, ymax, xmax = bbox
            (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                          ymin * frame.shape[0], ymax * frame.shape[0])
            cv2.rectangle(frame, (int(left), int(top)),
                          (int(right), int(bottom)), (255, 0, 0), 2)
            boxlabel = f'{label}: {int(confidence * 100)}%'
            cv2.putText(frame, boxlabel, (int(left), int(top) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)

        #face detection and recognition using face mesh landmarks
        recognized_faces = detect_and_recognize_faces_with_mesh(frame, face_mesh, known_faces_landmarks)
        
        
        #print(recognized_faces) # testing what the output of this is if it outputs names of known faces or not 
        
        
        for face_landmarks, name in recognized_faces:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            frame = detect_and_draw_face_mesh(frame, face_mesh, face_landmarks)
            cv2.putText(frame, name, (int(face_landmarks.landmark[0].x * frame.shape[1]), int(face_landmarks.landmark[0].y * frame.shape[0]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        
        #display the result
        #cv2.imshow('ARGUS object and face recognition', frame)

        
        if time.time() - start_time > duration: # what this line is supposed to do is 10 seconds after camera is opened it should break the loop
            break 
        
        # 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    #cv2.destroyAllWindows()
    
    # Extract only recognized names, skip "Unknown"
    recognized_names = [name for _, name in recognized_faces if name != "Unknown"]

    # Combine all recognized things
    things_seen = list(detected_labels.union(set(recognized_names)))

    return things_seen
    
    #return list(detected_labels), [name for _, name in recognized_faces]

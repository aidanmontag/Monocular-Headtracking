import cv2
import numpy as np
import time
import math
from ultralytics import YOLO
from pythonosc.udp_client import SimpleUDPClient
from collections import deque

# Load the YOLOv8n-pose model
model = YOLO("yolov8n-pose.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Set Camera FOV as a constant
CAMERA_FOV = 69  # degrees 

# Define 3D model of the face (mm)
MODEL_POINTS = np.array([
    [0.0, 0.0, 0.0],             # Nose tip (origin)
    [-30.0, 35.0, -30.0],        # Right eye
    [30.0, 35.0, -30.0],         # Left eye
    [-60.0, 20.0, -95.0],        # Right ear
    [60.0, 20.0, -95.0],         # Left ear
], dtype=np.float32)

# Setup up OSC Client with Target IP Address and Port
osc_client = SimpleUDPClient("127.0.0.1", 9000) #set to localhost for single node use case

# Get image dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(frame_height)

# Calculate the camera matrix
focal_length = frame_width / (2 * np.tan(np.deg2rad(CAMERA_FOV / 2)))
center = (frame_width / 2, frame_height / 2)
camera_matrix = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype=np.float32)

# Apply lens Distorion Correction (assuming none)
dist_coeffs = np.zeros((4, 1), dtype=np.float32)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get facial keypoints from yolo
    results = model(frame, verbose=False)
    
    # Draw the pose on the frame
    annotated_frame = results[0].plot()
    
    if len(results[0].keypoints.data) > 0 and results[0].keypoints.conf is not None:
        # Get the first person detected
        kp = results[0].keypoints.data[0]
        confidences = results[0].keypoints.conf[0].cpu().numpy()
        
        # Extract keypoints of interest (nose, eyes, ears)
        try:
            # Only use points with high enough confidence 
            CONFIDENCE_THRESHOLD = 0.5
            
            # Set keypoint position to none if not past confidence threshold
            nose = kp[0].cpu().numpy() if confidences[0] > CONFIDENCE_THRESHOLD else None
            right_eye = kp[2].cpu().numpy() if confidences[2] > CONFIDENCE_THRESHOLD else None
            left_eye = kp[1].cpu().numpy() if confidences[1] > CONFIDENCE_THRESHOLD else None
            right_ear = kp[4].cpu().numpy() if confidences[4] > CONFIDENCE_THRESHOLD else None
            left_ear = kp[3].cpu().numpy() if confidences[3] > CONFIDENCE_THRESHOLD else None
            
            valid_model_points = []
            valid_image_points = []
            
            #create array of 2D keypoints
            if confidences[4] > confidences[3]: 
                keypoints = [(nose, 0), (left_eye, 1), (right_eye, 2), (right_ear, 4)]

            else:
                keypoints = [(nose, 0), (left_eye, 1), (right_eye, 2), (left_ear, 3)]
            
            # Only keep neccesary data
            for point, idx in keypoints:
                if point is not None:
                    valid_model_points.append(MODEL_POINTS[idx])
                    valid_image_points.append(point[:2])  # Only x,y coordinates
            
            # Veify we have enough keypoints for PnP (4)
            if len(valid_image_points) >= 4:
                valid_model_points = np.array(valid_model_points, dtype=np.float32)
                valid_image_points = np.array(valid_image_points, dtype=np.float32)
                
                # Solve for pose
                success, rotation_vector, translation_vector = cv2.solvePnP(
                    valid_model_points, # 3D points of head model
                    valid_image_points, # 2D keypoints from image
                    camera_matrix,
                    dist_coeffs, 
                    flags=cv2.SOLVEPNP_P3P 
                )
                
                if success:
                    # Convert rotation vector to Euler angles
                    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                    pose_matrix = np.hstack((rotation_matrix, translation_vector))
                    # Add row [0,0,0,1] to make it a proper 4x4 transform matrix
                    pose_matrix = np.vstack((pose_matrix, [0, 0, 0, 1]))
                    
                    # Extract euler angles (in degrees)
                    proj_matrix = pose_matrix[:3, :]
                    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
                    pitch, yaw, roll = euler_angles.flatten()
                    
                    # Extract position (in mm)
                    x, y, z = translation_vector.flatten()
                    
                    # Display values
                    cv2.putText(
                        annotated_frame,
                        f"X: {x:.1f}, Y: {y:.1f}, Z: {z:.1f}", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (0, 255, 0), 
                        2
                    )
                    cv2.putText(
                        annotated_frame,
                        f"Pitch: {pitch:.1f}, Yaw: {yaw:.1f}, Roll: {roll:.1f}", 
                        (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (0, 255, 0), 
                        2
                    )
                    #Translate CV to UE
                    UEx = z
                    UEy = x
                    UEz = y

                    #add yaw offset
                    yaw_offset = math.degrees(math.atan2(UEy, UEx))

                    print(yaw_offset)

                    yaw = yaw + yaw_offset * 2
                    
                    # Send OSC messages to Unreal Engine
                    osc_client.send_message(
                        "/LiveLink/headtracker/locrot", 
                        [float(UEx), float(UEy), float(UEz), float(roll), float(pitch), float(yaw)]
                    )
                
        except (IndexError, cv2.error) as e:
            print(f"Error in pose estimation: {e}")

    # Show the frame
    cv2.imshow("Head Pose Estimation", annotated_frame)
    
    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
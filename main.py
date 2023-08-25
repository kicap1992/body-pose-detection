import cv2
import mediapipe as mp
import os
import numpy as np

# Create the assets directory if it doesn't exist
if not os.path.exists('assets'):
    os.makedirs('assets')

# Load the pose detection model
with mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Get the index of the last saved pose dataset image
    count = len([name for name in os.listdir('assets') if name.endswith('.jpg')])
    # Capture frames from the webcam
    cap = cv2.VideoCapture(0)
    
    # Load saved pose images and store their landmarks and filenames
    saved_landmarks = []
    saved_filenames = []
    for i in range(count):
        filename = f'assets/pose_{i}.jpg'
        image = cv2.imread(filename)
        if image is not None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            if results.pose_landmarks:
                landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.pose_landmarks.landmark])
                saved_landmarks.append(landmarks)
                saved_filenames.append(filename)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert the image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the image and find the landmarks
        results = pose.process(image)
        # Draw the landmarks on the image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        
        # Compare the pose with saved pose dataset images
        highest_similarity = -1
        most_similar_filename = ""
        if results.pose_landmarks:
            detected_landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.pose_landmarks.landmark])
            
            for i, saved_landmark in enumerate(saved_landmarks):
                # Calculate cosine similarity between the landmarks
                similarity = np.dot(detected_landmarks.flatten(), saved_landmark.flatten()) / (np.linalg.norm(detected_landmarks) * np.linalg.norm(saved_landmark))
                
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    most_similar_filename = saved_filenames[i]
        
        # Calculate similarity percentage
        similarity_percentage = round(highest_similarity * 100, 2)
        
        # Display the most similar filename and similarity percentage if similarity is above 96%
        if similarity_percentage > 94.6:
            text = f"Most Similar: {most_similar_filename} - Similarity: {similarity_percentage}%"
            cv2.putText(image, text, (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the image
        cv2.imshow('Pose Detection', image)

        # Save the image if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            filename = f'assets/pose_{count}.jpg'
            # Check if the file already exists and increment the count if it does
            while os.path.exists(filename):
                count += 1
                filename = f'assets/pose_{count}.jpg'
            cv2.imwrite(filename, image)
            print(f'Saved pose dataset image: {filename}')
            count += 1
        # Exit the loop if the 'ESC' key is pressed
        if cv2.waitKey(1) == 27:
            break

    # Release the capture and destroy the window
    cap.release()
    cv2.destroyAllWindows()

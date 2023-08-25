import cv2
import mediapipe as mp
import os
import time

# Create the assets directory if it doesn't exist
if not os.path.exists('assets'):
    os.makedirs('assets')

# Get the index of the last saved pose dataset image
count = len([name for name in os.listdir('assets') if name.endswith('.jpg')])

with mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert the image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        # Process the image and find the landmarks
        results = pose.process(image)
        # Draw the landmarks on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        # Display the image in a window
        cv2.imshow('Pose Detection', image)
        # Save the image and close the window if 5 seconds have passed
        if time.time() - start_time > 5:
            filename = f'assets/pose_{count}.jpg'
            # Check if the file already exists and increment the count if it does
            while os.path.exists(filename):
                count += 1
                filename = f'assets/pose_{count}.jpg'
            cv2.imwrite(filename, image)
            print(f'Saved pose dataset image: {filename}')
            count += 1
            # Close the window
            cv2.destroyAllWindows()
            break
        # Exit the loop if the 'ESC' key is pressed
        if cv2.waitKey(1) == 27:
            break
    # Release the capture and destroy the window
    cap.release()
    cv2.destroyAllWindows()
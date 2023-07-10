import cv2
import time
import numpy as np
import mediapipe as mp
from djitellopy import Tello

# Initialize Tello drone
tello = Tello()
tello.connect()

# Set up mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Variables for hand gesture control
palm_center = None
previous_gesture = None
gesture_mapping = {
    'ONE'  : 'takeoff',
    'TWO'  : 'land',
    'THREE': 'flip',
    'FOUR' : 'forward',
    'FIVE' : 'backward'
}

def perform_gesture_action(gesture):
    action = gesture_mapping.get(gesture)
    if action:
        if action == 'takeoff':
            tello.takeoff()
        elif action == 'land':
            tello.land()
        elif action == 'flip':
            tello.flip()
        elif action == 'forward':
            tello.move_forward(30)
        elif action == 'backward':
            tello.move_back(30)

# Main loop
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while True:
        # Read frame from camera
        ret, frame = tello.get_frame_read().frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process hand landmarks
        results = hands.process(image)

        # Draw hand landmarks on frame
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get palm center coordinates
                palm_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * frame.shape[1])
                palm_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * frame.shape[0])
                palm_center = (palm_x, palm_y)
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Detect hand gesture
        if palm_center is not None:
            # Calculate distance from palm center to thumb tip
            thumb_tip_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * frame.shape[1])
            thumb_tip_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * frame.shape[0])
            thumb_tip = (thumb_tip_x, thumb_tip_y)
            distance = np.linalg.norm(np.array(palm_center) - np.array(thumb_tip))

            # Classify gesture based on distance threshold
            if distance < 50:
                gesture = 'FIST'
            elif distance < 100:
                gesture = 'ONE'
            elif distance < 150:
                gesture = 'TWO'
            elif distance < 200:
                gesture = 'THREE'
            elif distance < 250:
                gesture = 'FOUR'
            else:
                gesture = 'FIVE'

            # Check if gesture has changed
            if gesture != previous_gesture:
                perform_gesture_action(gesture)
                previous_gesture = gesture

        # Display frame
        cv2.imshow('Tello Gesture Control', image)

        # Check for keypress to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Clean up
cv2.destroyAllWindows()
tello.end()

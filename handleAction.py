

import cv2
import mediapipe as mp
import pyautogui
import threading

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 24)  # High FPS for less lag

# State management
current_action = None
action_lock = threading.Lock()


# Key control functions
def press_key(key):
    pyautogui.keyDown(key)


def release_key(key):
    pyautogui.keyUp(key)


# Function to handle game control
def control_game(gesture):
    global current_action
    with action_lock:
        if gesture == "open_palm" and current_action != "accelerate":
            release_key('left')  # Ensure brake is not pressed
            press_key('right')
            current_action = "accelerate"
            print("Accelerating (Right Arrow)")

        elif gesture == "closed_fist" and current_action != "brake":
            release_key('right')  # Ensure acceleration is not pressed
            press_key('left')
            current_action = "brake"
            print("Braking (Left Arrow)")

        elif gesture == "neutral" and current_action is not None:
            release_key('right')
            release_key('left')
            current_action = None
            print("Neutral (Stop Actions)")


# Function to recognize hand gestures
def recognize_gesture(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    wrist = landmarks[0]

    # Calculate distances of fingertips to the wrist
    def distance(point1, point2):
        return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5

    index_dist = distance(index_tip, wrist)
    middle_dist = distance(middle_tip, wrist)
    ring_dist = distance(ring_tip, wrist)
    pinky_dist = distance(pinky_tip, wrist)
    thumb_dist = distance(thumb_tip, wrist)

    # Thresholds for gesture detection
    open_threshold = 0.2
    closed_threshold = 0.1

    # Open palm: All fingers extended
    if (index_dist > open_threshold and middle_dist > open_threshold and
            ring_dist > open_threshold and pinky_dist > open_threshold and
            thumb_dist > open_threshold):
        return "open_palm"

    # Closed fist: All fingers close to the wrist
    elif (index_dist < closed_threshold and middle_dist < closed_threshold and
          ring_dist < closed_threshold and pinky_dist < closed_threshold):
        return "closed_fist"

    return "neutral"


# Main camera loop
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = recognize_gesture(hand_landmarks.landmark)
            control_game(gesture)

    cv2.imshow('Hill Climbing Game Control', image)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()
release_key('right')
release_key('left')
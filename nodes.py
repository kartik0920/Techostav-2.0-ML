import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Initialize variables to store landmark values
        left_hand_landmarks = None
        right_hand_landmarks = None

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Determine if the hand is left or right
                hand_label = handedness.classification[0].label

                # Draw landmarks and connections
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Store landmarks for left or right hand
                if hand_label == "Left":
                    left_hand_landmarks = hand_landmarks
                elif hand_label == "Right":
                    right_hand_landmarks = hand_landmarks

        # Display left hand landmarks on the left bottom of the screen
        if left_hand_landmarks:
            y_offset = 30
            for idx, landmark in enumerate(left_hand_landmarks.landmark):
                landmark_text = f"Left {idx}: ({landmark.x:.2f}, {landmark.y:.2f}, {landmark.z:.2f})"
                cv2.putText(image, landmark_text, (10, image.shape[0] - y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                y_offset += 20

        # Display right hand landmarks on the right bottom of the screen
        if right_hand_landmarks:
            y_offset = 30
            for idx, landmark in enumerate(right_hand_landmarks.landmark):
                landmark_text = f"Right {idx}: ({landmark.x:.2f}, {landmark.y:.2f}, {landmark.z:.2f})"
                text_width = cv2.getTextSize(landmark_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
                cv2.putText(image, landmark_text, (image.shape[1] - text_width - 10, image.shape[0] - y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                y_offset += 20

        # Show the image with landmarks and text
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()





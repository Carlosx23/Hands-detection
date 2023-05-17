import cv2
import mediapipe as mp

# Import the necessary modules from Mediapipe for hand tracking
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Open the video capture device (webcam)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Initialize the hand tracking module
with mp_hands.Hands(
    static_image_mode=False,          # Set static_image_mode to False for video input
    max_num_hands=2,                  # Maximum number of hands to detect
    min_detection_confidence=0.3) as hands:   # Minimum confidence value for hand detection

    while True:
        # Read each frame from the video capture device
        ret, frame = cap.read()

        # If there's no frame, break out of the loop
        if ret == False:
            break

        # Get the height and width of the frame
        height, width, _ = frame.shape

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Convert the frame from BGR to RGB color space
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hands
        results = hands.process(frame_rgb)

        # If hands are detected in the frame
        if results.multi_hand_landmarks is not None:
            # Iterate through each detected hand
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the landmarks and connections on the frame
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,255), thickness=3, circle_radius=5),
                    mp_drawing.DrawingSpec(color=(255,0,255), thickness=4, circle_radius=5))

        # Show the frame with landmarks
        cv2.imshow('Frame',frame)

        # If 'Esc' key is pressed, exit the loop
        if cv2.waitKey(1) & 0xFF == 27:
            break

# Release the video capture device and close all windows
cap.release()
cv2.destroyAllWindows()

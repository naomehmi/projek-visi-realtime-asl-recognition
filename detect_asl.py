import cv2
import tensorflow as tf
import numpy as np
import mediapipe as mp

# Load the trained ASL model
model = tf.keras.models.load_model("models/asl_model_tfrecord_v4.keras")  # Change path if needed

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Define ASL Labels (A-Z)
asl_labels = [chr(i) for i in range(65, 91)]  # A-Z

# Start webcam capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip image for mirror effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get hand bounding box
            x_min = w
            y_min = h
            x_max = 0
            y_max = 0

            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x, x_min)
                y_min = min(y, y_min)
                x_max = max(x, x_max)
                y_max = max(y, y_max)

            # Expand bounding box slightly
            margin = 20
            x_min = max(x_min - margin, 0)
            y_min = max(y_min - margin, 0)
            x_max = min(x_max + margin, w)
            y_max = min(y_max + margin, h)

            # Crop and preprocess hand image
            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.shape[0] == 0 or hand_img.shape[1] == 0:
                continue

            hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            hand_img = cv2.resize(hand_img, (64, 64)) / 255.0  # Resize to match model input, Normalize
            hand_img = np.expand_dims(hand_img, axis=-1)  # Add channel dimension
            hand_img = np.expand_dims(hand_img, axis=0)  # Add batch dimension

            # Predict ASL letter
            prediction = model.predict(hand_img)
            predicted_index = np.argmax(prediction)
            predicted_letter = asl_labels[predicted_index]

            # Display prediction
            cv2.putText(frame, f"Predicted: {predicted_letter}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show frame
    cv2.imshow("ASL Hand Tracking", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

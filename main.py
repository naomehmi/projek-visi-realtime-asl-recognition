import cv2
import numpy as np
import tensorflow as tf
from preprocess import load_images

# Load trained model
model = tf.keras.models.load_model('models/asl_model.h5')

# Load label map
_, _, label_map = load_images('dataset/Train')
reverse_label_map = {v: k for k, v in label_map.items()}  # Reverse lookup

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        break

    # Convert frame to grayscale & preprocess
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64)) / 255.0  # Normalize
    input_img = np.expand_dims(resized, axis=[0, -1])  # Reshape for model

    # Make prediction
    prediction = model.predict(input_img)
    predicted_class = reverse_label_map[np.argmax(prediction)]  # Get class name

    # Display prediction on screen
    cv2.putText(frame, f'Prediction: {predicted_class}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ASL Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()

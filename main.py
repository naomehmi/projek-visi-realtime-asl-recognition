import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("models/asl_model_tfrecord.h5")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64)) / 255.0
    input_img = np.expand_dims(resized, axis=[0, -1])

    prediction = model.predict(input_img)
    predicted_class = chr(np.argmax(prediction) + ord('A'))  # Convert index to letter

    cv2.putText(frame, f'Prediction: {predicted_class}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ASL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

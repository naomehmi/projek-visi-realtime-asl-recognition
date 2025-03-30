import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from preprocess import load_tfrecord_dataset

# Load dataset
train_dataset = load_tfrecord_dataset("dataset/train/Letters.tfrecord", batch_size=64)

# Define CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    MaxPooling2D(1,1),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(1,1),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(26, activation='softmax')  # 26 output classes for ASL A-Z
])

# Compile & train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, epochs=15)

# Save model
model.save("models/asl_model_tfrecord_v4.keras")
print("Model trained and saved successfully!")

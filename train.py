import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from preprocess import load_tfrecord_dataset

# Load dataset
train_dataset = load_tfrecord_dataset("dataset/train/Letters.tfrecord", batch_size=32)

# Define CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(26, activation='softmax')  # 26 output classes for ASL A-Z
])

# Compile & train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, epochs=10)

# Save model
model.save("models/asl_model_tfrecord.h5")
print("Model trained and saved successfully!")

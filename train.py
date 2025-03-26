import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from preprocess import load_images

# Load dataset
train_images, train_labels, label_map = load_images('dataset/train')

# Reshape for CNN input
train_images = train_images.reshape(-1, 64, 64, 1)

# Model definition
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(label_map), activation='softmax')
])

# Compile & train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10, validation_split=0.2)

# Save model
model.save('models/asl_model.h5')
print("Model saved successfully!")

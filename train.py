import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from preprocess import load_tfrecord_dataset

# Load dataset
train_dataset = load_tfrecord_dataset("dataset/train/Letters.tfrecord", batch_size=32)

# Define MobileNetV2 model
base_model = MobileNetV2(input_shape=(64, 64, 3), include_top=False, weights='imagenet')

# Freeze base layers (optional for fine-tuning)
base_model.trainable = False  # Set to True if you want to fine-tune

# Add classification layers
inputs = Input(shape=(64, 64, 3))  # Ensure it's 3 channels
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(26, activation='softmax')(x)  # 26 ASL letters

# Create final model
model = Model(inputs, outputs)

# Compile & Train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, epochs=10)

# Save model
model.save("models/asl_mobilenetv2.h5")
print("MobileNetV2 Model trained and saved successfully!")

import tensorflow as tf
from preprocess import parse_tfrecord
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("models/asl_model_tfrecord_v4.keras")

# Function to load the test dataset
def load_test_dataset(tfrecord_path, batch_size=32):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord)  # Decode and preprocess
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

# Load test dataset
test_dataset = load_test_dataset("dataset/test/Letters.tfrecord", batch_size=64)

# Evaluate the model
loss, accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

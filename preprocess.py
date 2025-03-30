import tensorflow as tf

# Define feature description
feature_description = {
    'image/encoded': tf.io.FixedLenFeature([], tf.string),  # Image stored as bytes
    'image/object/class/label': tf.io.FixedLenFeature([], tf.int64),   # Label stored as int,
}

def load_tfrecord_dataset(tfrecord_path, batch_size=32):
    """Load dataset from TFRecord file"""
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord)  # Decode each example
    augmented_dataset = dataset.map(augment_image)
    # Combine original & augmented dataset
    final_dataset = dataset.concatenate(augmented_dataset)
    final_dataset = final_dataset.shuffle(1000).batch(batch_size)
    return final_dataset

# Function to parse TFRecord (Modify based on your TFRecord structure)
def parse_tfrecord(example_proto):
    feature_description = {
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/object/class/label": tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)

    # Decode image
    image = tf.image.decode_jpeg(parsed["image/encoded"], channels=3)
    image = tf.image.resize(image, [64, 64]) / 255.0  # Normalize
    image = tf.image.rgb_to_grayscale(image)

    label = parsed["image/object/class/label"] - 1
    return image, label

# Function to apply augmentations
def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)  # Flip horizontally
    image = tf.image.random_brightness(image, max_delta=0.5)  # Random brightness
    image = tf.image.random_contrast(image, 0.5, 1.5)  # Random contrast
    return image, label
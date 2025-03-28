import tensorflow as tf

# Define feature description
feature_description = {
    'image/encoded': tf.io.FixedLenFeature([], tf.string),  # Image stored as bytes
    'image/object/class/label': tf.io.FixedLenFeature([], tf.int64),   # Label stored as int,
}

def _parse_function(example_proto):
    """Parse TFRecord example"""
    example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_jpeg(example['image/encoded'], channels=1)
    image = tf.image.resize(image, (64, 64))
    image = tf.cast(image, tf.float32) / 255.0  
    label = tf.cast(example['image/object/class/label'], tf.int32) - 1
    return image, label

def load_tfrecord_dataset(tfrecord_path, batch_size=32):
    """Load dataset from TFRecord file"""
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(_parse_function)  # Decode each example
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

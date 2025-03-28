# to check feature keys and values from dataset
import tensorflow as tf

# Change this to your actual TFRecord file path
tfrecord_path = "dataset/train/Letters.tfrecord"

raw_dataset = tf.data.TFRecordDataset(tfrecord_path)

for raw_record in raw_dataset.take(1):  # Read the first record
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)

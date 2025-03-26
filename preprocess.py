import cv2;
import os;
import numpy as np;

def load_images(folder, img_size=(64,64)):
    images, labels = [], []
    classes = sorted(os.listdir(folder))
    label_map = { class_name: idx for idx, class_name in enumerate(classes) }

    for class_name in classes:
        class_folder = os.path.join(folder, class_name)
        for img_name in os.listdir(class_folder):
            img = cv2.imread(os.path.join(class_folder, img_name), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, img_size) / 255.0  # Normalize
            images.append(img)
            labels.append(label_map[class_name])

    return np.array(images), np.array(labels), label_map
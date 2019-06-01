
import os
import tensorflow as tf
import pathlib

#get current working directory
cwd = str(os.getcwd())

path_to_origin = "file://localhost" + cwd + "/train_data.tar.xz"
data_root_orig = tf.keras.utils.get_file(origin = path_to_origin, fname = "train_data", untar = True)
data_root = pathlib.Path(data_root_orig)

import random

all_image_paths = list(data_root.glob("*/*"))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

label_names = sorted(item.name for item in data_root.glob("*/") if item.is_dir())

label_to_index = dict((name, index) for index, name in enumerate(label_names))
all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]

def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels = 3)
    image = tf.image.resize(image, [256, 256])
    image /= 255.0
    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)

AUTOTUNE = tf.data.experimental.AUTOTUNE
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls = AUTOTUNE)

label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

#image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))


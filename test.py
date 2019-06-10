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
print(all_image_paths[0])

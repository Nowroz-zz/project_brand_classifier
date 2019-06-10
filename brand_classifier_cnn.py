
import os
import tensorflow as tf
import pathlib

# Obtains the current working directory and converts it into string
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
    image = tf.image.resize(image, [32, 32])
    image /= 255.0
    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)

AUTOTUNE = tf.data.experimental.AUTOTUNE
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls = AUTOTUNE)

label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))


# Training Parameters
learning_rate = 0.001
num_steps = 1000
batch_size = 10
image_count = 270


ds = image_label_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(batch_size)
ds = ds.prefetch(buffer_size=AUTOTUNE)

iter = ds.make_one_shot_iterator()

image_label_batch = iter.get_next()

image_batch = image_label_batch[0]
label_batch = image_label_batch[1]



image_X = image_batch
label_Y = label_batch

# Network Parameters
num_input = 32*32*3 # img shape: 32*32
num_classes = 3 # Classes: Apple, Lenovo and Samsung
dropout = 0.25 # Dropout, probability to drop a unit
is_training = True

# The Neural Network
# Reshapes Tensor input into 4-D: [Batch Size, Height, Width, Channel]
x = tf.reshape(image_X, shape = [-1, 32, 32, 3])

# Creates a Convolution Layer with 32 filters and a kernel size of 5
conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
# Max Pooling (down-sampling) with strides of 2 and kernel size of 2
conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

# Creates a Convolution Layer with 64 filters and a kernel size of 3
conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
# Max Pooling (down-sampling) with strides of 2 and kernel size of 2
conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

# Flattens the data to a 1-D vector for the fully connected layer
fc1 = tf.contrib.layers.flatten(conv2)

# Creates a Dense Layer with 1024 nodes
fc1 = tf.layers.dense(fc1, 1024)

# Applies Dropout (if is_training is False, dropout is not applied)
fc1 = tf.layers.dropout(fc1, rate = dropout, training = is_training)

# Final Output Layer, class prediction
out = tf.layers.dense(fc1, num_classes)

# logits
logits = out

# Defines loss
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = tf.cast(label_Y, dtype=tf.int32)))
    
# Optimizes the model
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)

train_op = optimizer.minimize(loss_op, global_step = tf.train.get_global_step())

# Creates a global variables initializer
init = tf.global_variables_initializer()

# Creates a saver object
saver = tf.train.Saver()

with tf.Session() as sess:

    # Check if the .data file exists
    exists = os.path.isfile('trained_model.data-00000-of-00001')
    
    if exists:
        print("******************************** A trained model already exists ********************************")

    else:
        # initializes all the variables
        sess.run(init)

        for step in range(num_steps):
            _, loss = sess.run([train_op, loss_op])
            print("..Step # %d-> loss - %f"%(step, loss))

        #saves the model by the name 'trained_model'
        saver.save(sess, 'trained_model')
        print("******************************** Finished training ********************************")


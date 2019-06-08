
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

def fetchData():
    ds = image_label_ds.apply(
    tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    iter = ds.make_one_shot_iterator()

    image_label_batch = iter.get_next()

    image_batch = image_label_batch[0]
    label_batch = image_label_batch[1]

    return image_batch, label_batch

X_train, Y_train = fetchData()

# Network Parameters
num_input = 32*32*3 # img shape: 32*32
num_classes = 3 # Classes: Apple, Lenovo and Samsung
dropout = 0.25 # Dropout, probability to drop a unit

# Create the neural network
# Tensor input become 4-D: [Batch Size, Height, Width, Channel]
x = tf.reshape(X_train, shape = [-1, 32, 32, 3])

# Convolution Layer with 32 filters and a kernel size of 5
conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
# Max Pooling (down-sampling) with strides of 2 and kernel size of 2
conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

# Convolution Layer with 64 filters and a kernel size of 3
conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
# Max Pooling (down-sampling) with strides of 2 and kernel size of 2
conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

# Flatten the data to a 1-D vector for the fully connected layer
fc1 = tf.contrib.layers.flatten(conv2)

# Fully connected layer (in tf contrib folder for now)
fc1 = tf.layers.dense(fc1, 1024)

# Apply Dropout (if is_training is False, dropout is not applied)
fc1 = tf.layers.dropout(fc1, rate = dropout, training = True)

# Output layer, class prediction
out = tf.layers.dense(fc1, num_classes)


# Define the model function (following TF Estimator Template)

# Build the neural network
# Because Dropout have different behavior at training and prediction time, we
# need to create 2 distinct computation graphs that still share the same weights.

logits_train = out
# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits_train, labels = tf.cast(Y_train, dtype=tf.int32)))
    

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)

train_op = optimizer.minimize(loss_op, global_step = tf.train.get_global_step())

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    #print(sess.run(tf.report_uninitialized_variables()))
    for step in range(num_steps):
        sess.run(train_op)
        print("Loss for step # %d: %f"%(step, sess.run(loss_op)))


##logits_test = conv_net(features, num_classes, dropout, reuse=True,is_training=False)

# Predictions
##pred_classes = tf.argmax(logits_test, axis=1)
##pred_probas = tf.nn.softmax(logits_test)

# for i in range(100):
# _ = sess.run(train_op, feed_dict={x: X_train, y: Y_train})

# accuracy = sess.run(pred_classes, feed_dict={x: X_test, y: Y_test})
# Evaluate the accuracy of the model
##acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)





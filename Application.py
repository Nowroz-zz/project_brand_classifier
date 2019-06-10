from tkinter import *
from tkinter import filedialog as fd
root = Tk()

import tensorflow as tf
import os

test_image_path = ""

def commOpenFile():
    global test_image_path
    image  = fd.askopenfile()
    test_image_path = image.name
    label = Label(text = test_image_path).pack()

def commRun():
    with tf.Session() as sess:

        # Check if the .data file exists
        exists = os.path.isfile('trained_model.data-00000-of-00001')

        if exists:
            
            def preprocess_image(image):
                image = tf.image.decode_jpeg(image, channels = 3)
                image = tf.image.resize(image, [32, 32])
                image /= 255.0
                return image

            def load_and_preprocess_image(path):
                image = tf.io.read_file(path)
                return preprocess_image(image)


            # Training Parameters
            learning_rate = 0.001
            num_steps = 1000
            batch_size = 10
            image_count = 270


            image_X = load_and_preprocess_image(test_image_path)

            # Network Parameters
            num_input = 32*32*3 # img shape: 32*32
            num_classes = 3 # Classes: Apple, Lenovo and Samsung
            dropout = 0.25 # Dropout, probability to drop a unit
            is_training = False

            # Create the neural network
            # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
            x = tf.reshape(image_X, shape = [-1, 32, 32, 3])

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
            fc1 = tf.layers.dropout(fc1, rate = dropout, training = is_training)

            # Output layer, class prediction
            out = tf.layers.dense(fc1, num_classes)



            logits = out

            # Creating a saver object
            saver = tf.train.Saver()

            global restore_count

            saver.restore(sess, tf.train.latest_checkpoint('./'))

            pred_class = tf.argmax(logits, axis = 1)
            pred_prob = tf.nn.softmax(logits)

            p_class, p_prob = sess.run([pred_class, pred_prob])


            label = Label(text = p_class).pack()

button = Button(text = "Run", width = 30, command = commRun).pack()
button = Button(text = "Browse Image", width = 30, command = commOpenFile).pack()


root.mainloop()

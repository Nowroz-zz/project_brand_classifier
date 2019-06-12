from tkinter import *
from tkinter import filedialog as fd
from tkinter import messagebox

root = Tk()
root.title("Brand Classifier")

import tensorflow as tf
import os
import numpy as np

test_image_path = ""

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
batch_size = 10

# Creates a placeholder to test the model
image_X = tf.placeholder(dtype = tf.float32)

# Network Parameters
num_input = 32*32*3 # img shape: 32*32
num_classes = 3 # Classes: Apple, Lenovo and Samsung
dropout = 0.25 # Dropout, probability to drop a unit
is_training = False

# The Neural Network
# Reshapes Tensor into 4-D: [Batch Size, Height, Width, Channel]
x = tf.reshape(image_X, shape = [-1, 32, 32, 3])

# Creates a Convolution Layer with 32 filters and a kernel size of 5
conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
# Max Pooling (down-sampling) with strides of 2 and kernel size of 2
conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

# Creates a Convolution Layer with 64 filters and a kernel size of 3
conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
# Max Pooling (down-sampling) with strides of 2 and kernel size of 2
conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

# Flattens the data to a 1-D vector for the fully connected layer i.e., the dense layer
fc1 = tf.contrib.layers.flatten(conv2)

# Creates a dense layer with 1024 nodes
fc1 = tf.layers.dense(fc1, 1024)

# Applies Dropout (if is_training is False, dropout is not applied)
fc1 = tf.layers.dropout(fc1, rate = dropout, training = is_training)

# Final output layer
out = tf.layers.dense(fc1, num_classes)

# Logits
logits = out

# Creates a saver object
saver = tf.train.Saver()

sess = tf.Session()

# Restores the model
saver.restore(sess, tf.train.latest_checkpoint('./'))

def commOpenFile():
    global test_image_path
    image  = fd.askopenfile()
    test_image_path = image.name
    
    # Set the path to pathVar to update the text of pathLabel
    pathVar.set("Path: "+test_image_path)

def commRun():
    pred_class = tf.argmax(logits, axis = 1)
    pred_prob = tf.nn.softmax(logits)

    if test_image_path:
        test_data = sess.run(load_and_preprocess_image(test_image_path))
        
        pred, prob = sess.run([pred_class, pred_prob], feed_dict = {image_X: test_data})

        pred = pred[0]
        prob = np.reshape(a = prob, newshape = 3)
        prob = max(prob)

        pred_str = ""

        if pred == 0:
            pred_str = "Apple"
        
        elif pred == 1:
            pred_str = "Lenovo"

        else:
            pred_str = "Samsung"

        outVar.set("Output: %s\nAccuracy: %f"%(pred_str, prob))


    else:
        outVar.set("Output: MUST SELECT AN IMAGE FIRST")


def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        sess.close()
        root.destroy()


button = Button(text = "Browse Image", width = 30, command = commOpenFile).grid(row = 0, column = 0)
button = Button(text = "Run", width = 30, command = commRun).grid(row = 1, column = 0)

#Creates a Tkinter string variable to dynamically update the pathLabel
pathVar = StringVar()
pathVar.set("Path:")

# Creates a label that displays the path of the selected image
pathLabel = Label(textvariable = pathVar).grid(row = 3)

#Creates a Tkinter string variable to dynamically update the outLabel
outVar = StringVar()
outVar.set("Output: ")

# Creates a label to display output
outLabel = Label(textvariable = outVar, fg = "red").grid(row = 5)

# Invokes on_closing method upon closing the window
root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()

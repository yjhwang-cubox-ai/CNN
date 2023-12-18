import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import numpy as np

# fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import pandas as pd
import cv2
import os
from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread, imshow, subplots, show

data = np.load('data.npy', allow_pickle=True)

train = data[:37000]
test = data[37000:]

train_X = []
train_y = []
for x in train:
    train_X.append(x[0])
    train_y.append(x[1])

test_X = []
test_y = []
for x in test:
    test_X.append(x[0])
    test_y.append(x[1])

train_X = np.array(train_X)
train_y = np.array(train_y)

test_X = np.array(test_X)
test_y = np.array(test_y)

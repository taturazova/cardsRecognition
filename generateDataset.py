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


# image = imread('cards/w07.jpg')
# images = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# imshow(images[0])
# show()

# data_generator = ImageDataGenerator(rotation_range=90, brightness_range=(
#     0.2, 1.5), shear_range=15.0)
# data_generator.fit(images)
# image_iterator = data_generator.flow(images)

# plt.figure(figsize=(16, 16))
# for i in range(16):
#     plt.subplot(4, 4, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(image_iterator.next()[0].astype('int'))
# plt.show(S

data = []

for i, img in tqdm(enumerate(os.listdir("cards/"))):
    label = i
    img = cv2.imread("cards/" + img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (180, 180))
    imgs = img.reshape((1, img.shape[0], img.shape[1], 1))
    data_generator = ImageDataGenerator(
        rotation_range=90, brightness_range=(0.5, 1.5), shear_range=15.0
    )
    data_generator.fit(imgs)
    image_iterator = data_generator.flow(imgs)

    for x in range(750):
        img_transformed = image_iterator.next()[0].astype("int") / 255
        data.append([img_transformed, label])

shuffle(data)

np.save("data.npy", np.array(data, dtype=object))

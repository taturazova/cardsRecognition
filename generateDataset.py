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


# img = cv2.imread("cards/c01.jpg", cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img, (180, 180))
# imgs = img.reshape((1, img.shape[0], img.shape[1], 1))


# # Example of ImageDataGenerator for augmentation
# data_generator = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=False,
#     vertical_flip=False,
#     brightness_range=[0.8, 1.2],
# )
# data_generator.fit(imgs)
# image_iterator = data_generator.flow(imgs)

# plt.figure(figsize=(16, 16))
# for i in range(16):
#     plt.subplot(4, 4, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(image_iterator.next()[0].astype("int"))
# plt.show()

data = []
images = []

for i, img in tqdm(enumerate(os.listdir("cards/"))):
    label = i
    img = cv2.imread("cards/" + img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (180, 180))

    # Original image
    images.append(img)

# Add reversed cards to the dataset
reversed_images = []

for img in images:
    img_flipped = cv2.flip(img, 0)
    reversed_images.append(img_flipped)

images = images + reversed_images

for i, img in enumerate(images):
    label = i
    imgs = img.reshape((1, img.shape[0], img.shape[1], 1))

    data_generator = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        vertical_flip=False,
        brightness_range=[0.8, 1.2],
    )
    data_generator.fit(imgs)
    image_iterator = data_generator.flow(imgs)

    for x in range(750):
        img_transformed = image_iterator.next()[0].astype("int") / 255
        data.append([img_transformed, label])

shuffle(data)

np.save("data.npy", np.array(data, dtype=object))

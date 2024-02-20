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


img = cv2.imread("cards/c01.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (180, 180))
imgs = img.reshape((1, img.shape[0], img.shape[1], 1))

# Convert grayscale image to 3-channel image
img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

# Augmented images (rotated versions)
for angle in range(0, 360, 45):
    # Pad the image to ensure all pixels remain within the frame after rotation
    pad_value = int(np.min(img_rgb))  # Use the minimum pixel value for padding
    padded_img = cv2.copyMakeBorder(
        img_rgb, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=pad_value
    )
    rotated_img = np.array(
        ImageDataGenerator().apply_transform(padded_img, {"theta": angle})
    )
    imshow(rotated_img)
    show()

# data_generator = ImageDataGenerator(
#     rotation_range=180, brightness_range=(0.2, 1.5), shear_range=15.0
# )
# data_generator.fit(images)
# image_iterator = data_generator.flow(images)

# plt.figure(figsize=(16, 16))
# for i in range(16):
#     plt.subplot(4, 4, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(image_iterator.next()[0].astype("int"))
# plt.show()

# data = []

# for i, img in tqdm(enumerate(os.listdir("cards/"))):
#     label = i
#     img = cv2.imread("cards/" + img, cv2.IMREAD_GRAYSCALE)
#     img = cv2.resize(img, (180, 180))
#     imgs = img.reshape((1, img.shape[0], img.shape[1], 1))

#     # Original image
#     data.append([img / 255, label])

#     # Augmented images (rotated versions)
#     for angle in range(0, 360, 45):
#         rotated_img = np.array(
#             ImageDataGenerator().apply_transform(img, {"theta": angle})
#         )
#         data.append([rotated_img / 255, label])

# shuffle(data)

# np.save("data.npy", np.array(data, dtype=object))

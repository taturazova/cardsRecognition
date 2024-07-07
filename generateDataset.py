import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import cv2
import numpy as np
import os
from random import shuffle

import matplotlib.pyplot as plt


# img = cv2.imread("cards/m00.jpg", cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img, (250, 250))
# imgs = img.reshape((1, img.shape[0], img.shape[1], 1))

# # Example of ImageDataGenerator for augmentation
# data_generator = ImageDataGenerator(
#     rotation_range=180,
#     width_shift_range=0,
#     height_shift_range=0,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=False,
#     vertical_flip=False,
#     brightness_range=[0.1, 1.2],
# )
# data_generator.fit(imgs)
# image_iterator = data_generator.flow(imgs)

# plt.figure(figsize=(16, 16))
# for i in range(16):
#     plt.subplot(4, 4, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(image_iterator.__next__()[0].astype("int"))
# plt.show()

data = []
images = []

for i, img in tqdm(enumerate(os.listdir("cards/"))):
    label = i
    img = cv2.imread("cards/" + img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (180, 180))

    # Original image
    images.append(img)

for i, img in tqdm(enumerate(images)):
    label = i
    imgs = img.reshape((1, img.shape[0], img.shape[1], 1))

    data_generator = ImageDataGenerator(
        rotation_range=360,
        width_shift_range=0,
        height_shift_range=0,
        shear_range=0.2,
        zoom_range=0,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.2, 2],
    )
    data_generator.fit(imgs)
    image_iterator = data_generator.flow(imgs)

    for x in tqdm(range(750)):
        img_transformed = image_iterator.__next__()[0].astype("int") / 255
        for blur_size in range(3, 10, 2):
            img_blurred = cv2.GaussianBlur(img_transformed, (blur_size, blur_size), 0)
            data.append([img_blurred, label])
        data.append([img_transformed, label])


def generate_data_in_chunks():
    chunk_size = 1000
    total_data = len(data)
    for i in range(0, total_data, chunk_size):
        yield data[i : i + chunk_size]


shuffle(data)

np.save("data.npy", np.array(data, dtype=object))

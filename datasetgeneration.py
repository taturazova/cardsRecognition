import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import cv2
import numpy as np
import os
from random import shuffle

import matplotlib.pyplot as plt


# Function to get only the image out of the card, crop the borders
def crop_card_image(image_gray):
    image = cv2.resize(image_gray, (350, 600))

    # Apply Gaussian Blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(image, (7, 7), 0)

    # Define a sharpening kernel
    sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    # Apply the sharpening kernel to the grayscale image
    sharpened = cv2.filter2D(blurred, -1, sharpening_kernel)

    # Use Canny edge detection
    edges = cv2.Canny(sharpened, 150, 255)

    # Find contours on the edges detected image using cv2.CHAIN_APPROX_SIMPLE
    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    roi = None

    for i, h in enumerate(hierarchy[0]):

        # Use polygon approximation to simplify the contour
        epsilon = 0.02 * cv2.arcLength(contours[i], True)
        approx = cv2.approxPolyDP(contours[i], epsilon, True)
        area = cv2.contourArea(contours[i])

        # Calculate the length of the contour
        contour_length = cv2.arcLength(contours[i], True)

        # Calculate the convex hull of the contour
        hull_length = cv2.arcLength(cv2.convexHull(contours[i]), True)

        # Calculate the ratio of contour length to convex hull length
        ratio = contour_length / hull_length if hull_length > 0 else contour_length

        # Only check rectangles
        if ratio < 2 and len(approx) == 4 and area > 5000:

            rect = cv2.minAreaRect(approx)
            center, (w, h), angle = rect
            aspect_ratio = max(w, h) / min(w, h)

            # If aspect ratio fits the aspect ratio of a tarot card
            if 1.2 < aspect_ratio < 1.8:

                polygon_vertices = [point[0] for point in approx]

                # Determine the target width based on the maximum width of the base of the polygon
                target_width = max(
                    np.linalg.norm(polygon_vertices[1] - polygon_vertices[2]),
                    np.linalg.norm(polygon_vertices[3] - polygon_vertices[0]),
                )

                # Calculate the target height based on the aspect ratio
                target_height = int(target_width * aspect_ratio)

                # Define the target rectangle coordinates
                target_vertices = np.array(
                    [
                        [0, 0],
                        [0, target_height],
                        [target_width, target_height],
                        [target_width, 0],
                    ],
                    dtype=np.float32,
                )

                # Convert the polygon_vertices to NumPy array
                polygon_vertices = np.array(polygon_vertices, dtype=np.float32)

                # Compute the perspective transformation matrix
                matrix = cv2.getPerspectiveTransform(polygon_vertices, target_vertices)

                # Apply the perspective transformation
                roi = cv2.warpPerspective(
                    image, matrix, (int(target_width), int(target_height))
                )
                # Return roi
                return roi
    return roi


# Function to apply rotations to an image
def add_rotations(x):
    rotations = [0, 90, 180, 270]
    rotated_images = []
    for angle in rotations:
        rotated_image = np.rot90(x, k=angle // 90)
        rotated_images.append(rotated_image)
    return rotated_images


# Custom image data generator to include additional rotations
class CustomImageDataIterator:
    def __init__(self, imgs, **kwargs):
        self.imgs = imgs
        self.data_generator = ImageDataGenerator(**kwargs)
        self.data_generator.fit(self.imgs)
        self.image_iterator = self.data_generator.flow(self.imgs)

    def nextRotations(self):
        img = self.image_iterator.__next__()[0]
        return add_rotations(img)  # returns 4 rotations for the generated image


# img = cv2.imread("cards/m00.jpg", cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img, (250, 250))
# imgs = img.reshape((1, img.shape[0], img.shape[1], 1))

# # Example of ImageDataGenerator for augmentation

# image_iterator = CustomImageDataIterator(
#     imgs,
#     rotation_range=0,
#     width_shift_range=0,
#     height_shift_range=0,
#     shear_range=0.2,
#     zoom_range=0.1,
#     horizontal_flip=True,
#     vertical_flip=False,
#     brightness_range=[0.01, 1.5],
# )

# # Collect 36 images (4 images per call, 9 calls)
# images = []
# for _ in range(9):
#     rotations = image_iterator.nextRotations()
#     for img in rotations:
#         images.append(img)

# # Display the images in a 6x6 grid
# plt.figure(figsize=(16, 16))
# for i in range(36):
#     plt.subplot(6, 6, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(images[i].astype("int"))
# plt.show()

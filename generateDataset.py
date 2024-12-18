from tqdm import tqdm
import cv2
import numpy as np
import os
from random import shuffle


from datasetgeneration import CustomImageDataIterator, crop_card_image
import matplotlib.pyplot as plt


data_dir = "cards/"
chunk_size = 10
output_file = "data.npy"
N = 16
DEBUG = True


# N*20*78*2 images generated
def process_and_augment_images(image_directories, chunk_number, chunk_size):
    print("Generating image chunk")
    data = []
    images = []

    # Load and preprocess images
    for idx, directory in enumerate(image_directories):
        image_files = os.listdir(directory)
        label = chunk_number * chunk_size + idx

        for img_name in image_files:
            img_path = os.path.join(directory, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Crop only the card image, remove borders
            img_cropped = crop_card_image(img)

            img = cv2.resize(img if img_cropped is None else img_cropped, (180, 180))
            # if DEBUG:
            #     cv2.imshow("IMAGE", img)
            #     cv2.waitKey(0)
            images.append((img, label))

    # Process images
    for idx, img_tuple in enumerate(tqdm(images)):
        img = img_tuple[0]
        imgs = img.reshape((1, img.shape[0], img.shape[1], 1))
        label = img_tuple[1]

        image_iterator = CustomImageDataIterator(
            imgs,
            rotation_range=0.2,
            width_shift_range=0,
            height_shift_range=0,
            shear_range=0.2,
            zoom_range=0,
            horizontal_flip=True,
            vertical_flip=False,
            brightness_range=[0.01, 1.5],
        )

        if DEBUG and chunk_number == 0 and idx == 0:
            # Display the images in a 6x6 grid
            plt.figure(figsize=(16, 16))
            display_images = []

        # For each image, get N variations of rotation sets from the image_iterator = n*20 variations
        for i in range(N):
            image_transformed_rotations = image_iterator.nextRotations()

            # For each of the 4 rotations, add 4 blurred images + the rotation image itself = 20 images total
            for _, img_transformed in enumerate(image_transformed_rotations):
                img_transformed = img_transformed.astype("float32") / 255.0
                for blur_size in range(3, 10, 2):
                    img_blurred = cv2.GaussianBlur(
                        img_transformed, (blur_size, blur_size), 0
                    )
                    img_blurred = img_blurred[:, :, np.newaxis]

                    data.append([img_blurred, label])
                    if DEBUG and chunk_number == 0 and idx == 0:
                        display_images.append(img_transformed)
                        data.append([img_blurred, label])
                data.append([img_transformed, label])

                if DEBUG and chunk_number == 0 and idx == 0:
                    display_images.append(img_transformed)

        if DEBUG and chunk_number == 0 and idx == 0:
            for i in range(100):
                plt.subplot(10, 10, i + 1)
                plt.imshow(display_images[i])
                plt.axis("off")  # Hide the axes
            plt.tight_layout()
            plt.show()
    return data


def generate_data_in_chunks(image_directories, chunk_size):
    total_images = len(image_directories)
    for i in range(0, total_images, chunk_size):
        chunk_directories = image_directories[i : i + chunk_size]
        data_chunk = process_and_augment_images(
            chunk_directories, i // chunk_size, chunk_size
        )
        print(f"Generated chunk #{i}")
        yield data_chunk


def save_data_chunks(data, output_file):
    for chunk_idx, data_chunk in enumerate(data):
        print(f"Saving chunk # {chunk_idx}")
        if chunk_idx == 0:
            # Save first chunk as new file
            np.save(output_file, np.array(data_chunk, dtype=object))
        else:
            # Load existing data, append new chunk, and save again
            existing_data = np.load(output_file, allow_pickle=True)
            combined_data = np.concatenate(
                (existing_data, np.array(data_chunk, dtype=object))
            )
            np.save(output_file, combined_data)


# List all image files
image_directories = [f.path for f in os.scandir(data_dir) if f.is_dir()]
# shuffle(image_files)


# Generate data in chunks and save
data = generate_data_in_chunks(image_directories, chunk_size)
save_data_chunks(data, output_file)

# print(f"Data saved to {output_file}")


# data = []
# images = []

# for i, img in tqdm(enumerate(os.listdir("cards/"))):
#     label = i
#     img = cv2.imread("cards/" + img, cv2.IMREAD_GRAYSCALE)
#     img = cv2.resize(img, (180, 180))

#     # Original image
#     images.append(img)

# for i, img in tqdm(enumerate(images)):
#     label = i
#     imgs = img.reshape((1, img.shape[0], img.shape[1], 1))

#     image_iterator = CustomImageDataIterator(
#         imgs,
#         rotation_range=5,
#         width_shift_range=0,
#         height_shift_range=0,
#         shear_range=0.2,
#         zoom_range=0.1,
#         horizontal_flip=True,
#         vertical_flip=False,
#         brightness_range=[0.01, 1.5],
#     )

#     for x in tqdm(range(350)):
#         image_transformed_rotations = image_iterator.nextRotations()
#         for img_transformed in image_transformed_rotations:
#             img_transformed = img_transformed.astype("int") / 255
#             for blur_size in range(3, 10, 2):
#                 img_blurred = cv2.GaussianBlur(
#                     img_transformed, (blur_size, blur_size), 0
#                 )
#                 data.append([img_blurred, label])
#             data.append([img_transformed, label])


# def generate_data_in_chunks():
#     chunk_size = 1000
#     total_data = len(data)
#     for i in range(0, total_data, chunk_size):
#         yield data[i : i + chunk_size]


# shuffle(data)

# np.save("data.npy", np.array(data, dtype=object))

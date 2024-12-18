import cv2
import numpy as np
import json
from tensorflow import keras
import matplotlib.pyplot as plt

# Read card info
with open("card_data.json", "r") as file:
    card_list = json.load(file)


def resizeInputImage(image, new_height):
    # Get the original dimensions
    original_height, original_width = image.shape[:2]

    # Calculate the new width while maintaining the aspect ratio
    aspect_ratio = original_width / original_height
    new_width = int(new_height * aspect_ratio)

    # Resize the image
    return cv2.resize(image, (new_width, new_height))


# DETECTING SINGLE IMAGE
# Read Image
image = cv2.imread("testImages/3cards_3.jpg")
if image.shape[0] > 2000:
    image = resizeInputImage(image, 2000)

# convert the image to grayscale format
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Apply Gaussian Blur to reduce noise and improve edge detection
blurred = cv2.GaussianBlur(img_gray, (7, 7), 0)

# cv2.imshow("Blurred", cv2.resize(blurred, (0, 0), fx=0.5, fy=0.5))
# cv2.waitKey(0)

# Define a sharpening kernel
sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

# Apply the sharpening kernel to the grayscale image
sharpened = cv2.filter2D(blurred, -1, sharpening_kernel)

cv2.imshow("Sharpened", cv2.resize(sharpened, (0, 0), fx=0.5, fy=0.5))
cv2.waitKey(0)


# Use Canny edge detection
edges = cv2.Canny(sharpened, 150, 255)


# Find contours on the edges detected image using cv2.CHAIN_APPROX_SIMPLE
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
cv2.imshow("image", image)
cv2.waitKey(0)


detectedCard = None
cards = []

for i, h in enumerate(hierarchy[0]):
    # if h[3] == -1:  # contour has no parent

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

    # Create a blank image to draw the contour
    # if ratio < 2 and area > 1000:
    #     contour_image = np.zeros_like(image)
    #     cv2.drawContours(contour_image, contours, i, (0, 255, 0), 3)

    #     print(ratio, len(approx), area)
    #     # Display the result
    #     cv2.imshow(
    #         "Contour {}".format(i + 1),
    #         cv2.resize(contour_image, (0, 0), fx=0.2, fy=0.2),
    #     )
    #     cv2.waitKey(0)

    if ratio < 2 and len(approx) == 4 and area > 1000:

        rect = cv2.minAreaRect(approx)
        center, (w, h), angle = rect
        aspect_ratio = max(w, h) / min(w, h)

        polygon_vertices = [point[0] for point in approx]
        # Convert the polygon_vertices to NumPy array
        polygon_vertices = np.array(polygon_vertices, dtype=np.float32)

        polygon_area = cv2.contourArea(polygon_vertices)

        if 1.2 < aspect_ratio < 1.9:

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

            # Compute the perspective transformation matrix
            matrix = cv2.getPerspectiveTransform(polygon_vertices, target_vertices)

            # Apply the perspective transformation
            roi = cv2.warpPerspective(
                img_gray, matrix, (int(target_width), int(target_height))
            )

            detectedCard = (roi, polygon_area)
            cards.append((roi, polygon_vertices))

# Load the saved model from the .h5 file
model = keras.models.load_model("25epochs_conv_newdataset.keras")

cards.sort(key=lambda x: x[1][0][0])

# Classify cards
resulting_cards = []

for roi, _ in cards:
    roi = cv2.resize(roi, (180, 180))
    roi = roi.astype("float") / 255.0  # Normalize pixel values
    cv2.imshow("Image", roi)
    cv2.waitKey(0)

    roi = np.reshape(roi, (1, roi.shape[0], roi.shape[1], 1))
    # Make predictions using the loaded model
    predictions = model.predict(roi)

    # Decode the predictions (assuming your model output is categorical)
    class_index = np.argmax(predictions) % 78

    # Print the predicted class
    # print(
    #     f'Predicted Class: {card_list[(class_index % 78)]["name"]}\nCard Meaning: {card_list[class_index]["keywords"]}'
    # )
    resulting_cards.append(class_index)

# Filter cards, to remove duplicates and preserve order
resulting_cards = list(dict.fromkeys(resulting_cards))

for class_index in resulting_cards:
    print(
        f'Predicted Class: {card_list[(class_index % 78)]["name"]}\nCard Meaning: {card_list[class_index]["keywords"]}'
    )

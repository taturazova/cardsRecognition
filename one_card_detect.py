import cv2
import numpy as np
import json
from tensorflow import keras
import matplotlib.pyplot as plt

# Read card info
with open("card_data.json", "r") as file:
    card_list = json.load(file)

# Read Image
image = cv2.imread("testImages/pageofcups_light.jpg")
# convert the image to grayscale format
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# apply binary thresholding
ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)

# detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
contours, hierarchy = cv2.findContours(
    image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
)
# cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
# cv2.imshow("image", cv2.resize(image, (0, 0), fx=0.2, fy=0.2))
# cv2.waitKey(0)
roi = None
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
    ratio = contour_length / hull_length if hull_length > 0 else 1

    # if area > 5000:
    #     # Create a blank image to draw the contour
    #     contour_image = np.zeros_like(image)
    #     cv2.drawContours(contour_image, contours, i, (0, 255, 0), 3)

    #     print(ratio, len(approx), area)
    #     # Display the result
    #     cv2.imshow(
    #         "Contour {}".format(i + 1),
    #         cv2.resize(contour_image, (0, 0), fx=0.2, fy=0.2),
    #     )
    #     cv2.waitKey(0)

    if ratio < 2 and len(approx) == 4 and area > 5000:

        rect = cv2.minAreaRect(approx)
        center, (w, h), angle = rect
        aspect_ratio = max(w, h) / min(w, h)

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

cv2.imshow("image", roi)
cv2.waitKey(0)

roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
roi = cv2.resize(roi, (180, 180))
roi = roi.astype("float") / 255.0  # Normalize pixel values

roi = np.reshape(roi, (1, roi.shape[0], roi.shape[1], 1))

# Load the saved model from the .h5 file
model = keras.models.load_model("250epochs_conv.keras")

# Make predictions using the loaded model
predictions = model.predict(roi)

# Decode the predictions (assuming your model output is categorical)
class_index = np.argmax(predictions) % 78

# Print the predicted class
print(
    f'Predicted Class: {card_list[(class_index % 78)]["name"]}\nCard Meaning: {card_list[class_index]["keywords"]}'
)

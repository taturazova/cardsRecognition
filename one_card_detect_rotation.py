import cv2
import numpy as np
import json
from tensorflow import keras
import matplotlib.pyplot as plt

# Read card info
with open("card_data.json", "r") as file:
    card_list = json.load(file)

# Read Image
image = cv2.imread("testImages/thefool.jpg")
# convert the image to grayscale format
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# apply binary thresholding
ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)

# detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
contours, hierarchy = cv2.findContours(
    image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
)

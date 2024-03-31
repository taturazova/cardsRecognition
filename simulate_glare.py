import cv2
import numpy as np


def simulate_flash_glare(image, glare_intensity=0.7):
    # Create a circular mask
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    cv2.circle(
        mask,
        (image.shape[1] // 2, image.shape[0] // 2),
        min(image.shape[0], image.shape[1]) // 3,
        255,
        -1,
    )

    # Apply the mask to the original image
    glare_image = cv2.bitwise_and(image, image, mask=mask)

    # Apply the glare effect using Gaussian blur
    glare_image = cv2.GaussianBlur(glare_image, (0, 0), sigmaX=10)

    # Adjust the glare intensity
    glare_image = cv2.addWeighted(image, 1, glare_image, glare_intensity, 0)

    return glare_image


# Load an example card image
image_path = "cards/m00.jpg"
card_image = cv2.imread(image_path)

# Simulate flash glare
glare_intensity = 2  # Adjust the glare intensity as needed
image_with_glare = simulate_flash_glare(card_image, glare_intensity)

# Display the original and augmented images
cv2.imshow("Original Image", card_image)
cv2.imshow("Image with Glare", image_with_glare)
cv2.waitKey(0)
cv2.destroyAllWindows()

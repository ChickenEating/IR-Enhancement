import cv2
import numpy as np

def decomposition(image, num_scales=3):
    # Convert the image to a NumPy array
    image_np = np.array(image)

    # Initialize lists to store contrast and detail components
    contrast_components = []
    detail_components = []

    # Generate Gaussian pyramid
    pyramid = [image_np]
    for _ in range(num_scales - 1):
        image_np = cv2.pyrDown(image_np)
        pyramid.append(image_np)

    # Reconstruct contrast and detail components
    for i in range(num_scales - 1):
        contrast_component = cv2.pyrUp(pyramid[i + 1], dstsize=(pyramid[i].shape[1], pyramid[i].shape[0]))
        detail_component = pyramid[i] - contrast_component

        contrast_components.append(contrast_component)
        detail_components.append(detail_component)

    # The last scale's contrast component is the original image
    contrast_components.append(pyramid[-1])

    return contrast_components, detail_components

# # Example usage:
# image = cv2.imread("your_image.jpg", cv2.IMREAD_GRAYSCALE)
# contrast_components, detail_components = decomposition(image)

# # Accessing individual components (e.g., the second scale)
# second_scale_contrast = contrast_components[1]
# second_scale_detail = detail_components[1]

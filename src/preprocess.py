import cv2


def preprocess(images):
    processed_images = []
    for image in images:
        edges = cv2.Canny(image, 16, 100)
        processed_images.append(edges)
    return processed_images

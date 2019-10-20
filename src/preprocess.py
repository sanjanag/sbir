import cv2


def preprocess(images):
    print("Applying grayscale transformation")
    processed_images = convert_grayscale(images)
    return processed_images


def get_edge_imsage(images):
    edge_images = []
    for image in images:
        edges = cv2.Canny(image, 16, 100)
        edge_images.append(edges)
    return edge_images

def convert_grayscale(images):
    gray_images = []
    for img in images:
        gray_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return gray_images

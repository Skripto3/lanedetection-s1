import cv2
import numpy as np


def canny(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(
        image, (7, 7), 0# (5,5) oder (7,7) je nachdem wie stark der filter sein soll (je größer, desto stärker)
    )
    canny = cv2.Canny(blur, 100, 150)
    return canny


def image_overlay(image, overlay):
    return cv2.addWeighted(image, 0.8, overlay, 1, 1)


def white_image(image):
    avr_brightness = gat_avrage_brightness(image)
    blur = bilateral_filter(avr_brightness)

    l_thresh = int(np.percentile(avr_brightness, 99))
    l_thresh = np.clip (l_thresh, 120, 210)

    _, white_mask = cv2.threshold(
        blur, l_thresh, 255, cv2.THRESH_BINARY
    )


    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = np.abs(sobelx)

    max_val = np.max(sobelx)
    if max_val > 0:
        sobel_norm = sobelx / max_val
    else:
        sobel_norm = sobelx

    grad_weight = np.clip((sobel_norm - 0.18) / 0.9, 0.35, 1.0)
    grad_mask = np.uint8(255 * grad_weight)

    combined = cv2.bitwise_and(white_mask, grad_mask)

    combined_image = cv2.bitwise_and(image, image, mask=combined)
    return combined_image


def gat_avrage_brightness(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    brightness = hls[:, :, 1]
    return brightness

def bilateral_filter(image):
    return cv2.bilateralFilter(image, 7, 50, 50)

def morphological_cleanup(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    cleaned_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return cleaned_image

def clean_small_objects(image, min_size=300):

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        image, connectivity=8
    )

    clean = np.zeros_like(image)

    for i in range(1, num_labels):  # überspringe Hintergrund
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_size:
            clean[labels == i] = 255
    return clean

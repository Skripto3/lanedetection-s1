import cv2
import numpy as np


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # blur = cv2.GaussianBlur(
    #     gray, (7, 7), 0# (5,5) oder (7,7) je nach dem wie stark der filter sein soll (je größer desto stärker)
    # )
    canny = cv2.Canny(gray, 50, 300)
    return canny


def image_overlay(image, overlay):
    return cv2.addWeighted(image, 0.8, overlay, 1, 1)


def white_image(image):
    avr_brightness = gat_avrage_brightness(image)

    l_thresh = int(np.percentile(avr_brightness, 70))
    l_thresh = np.clip (l_thresh, 120, 210)

    _, white_mask = cv2.threshold(
        avr_brightness, l_thresh, 255, cv2.THRESH_BINARY
    )

    white_image = cv2.bitwise_and(image, image, mask=white_mask)
    return white_image


def gat_avrage_brightness(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    brightness = hls[:, :, 1]
    return brightness

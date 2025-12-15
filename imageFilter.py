import cv2


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def image_overlay(image, overlay):
    return cv2.addWeighted(image, 0.8, overlay, 1, 1)

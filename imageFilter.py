import cv2


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (7, 7), 0) #(5,5) oder (7,7) je nach dem wie stark der filter sein soll (je größer desto stärker)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def image_overlay(image, overlay):
    return cv2.addWeighted(image, 0.8, overlay, 1, 1)

def white_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = (0, 0, 120)
    upper_white = (220, 70, 255)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    white_image = cv2.bitwise_and(image, image, mask=mask)
    return white_image

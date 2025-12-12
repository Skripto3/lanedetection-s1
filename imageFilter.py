import cv2


def canny(image):
    '''
    Docstring für canny

    :param image: Beschreibung

    :return: imma as canny (1channel)
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

import cv2
import numpy as np

def region_of_interest(image):
    '''
    Erstellt eine Maske, um nur den Bereich der Straße zu fokussieren.
    Kompatibel mit Graustufenbildern (Canny).

    return: leck_eier
    '''

    height = image.shape[0]
    width = image.shape[1]

    # Punkte werden relativ zur Bildgröße berechnet
    polygons = np.array([
        [
            (int(width * 0.1), height),             # Unten links
            (int(width * 0.95), height),            # Unten rechts
            (int(width * 0.6), int(height * 0.6)),  # Oben rechts
            (int(width * 0.4), int(height * 0.6))   # Oben links
        ]
    ], np.int32)

    # Erstellt schwarze Maske
    mask = np.zeros_like(image)

    # Füllt Trapez weiß (255)
    cv2.fillPoly(mask, polygons, 255)

    # Pixel behalten die innerhalb Maske
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image

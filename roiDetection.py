import cv2
import numpy as np

import imageFilter as iF


#Dummy class to test
def region_of_interest(frame):
    #obere hälfte des bildes schwarzen

    frame[:int(frame.shape[0] / 3), :] = 0

    return frame

import cv2
import numpy as np

import imageFilter as iF
import roiDetection as rD


def lane_detection(input_path):
    '''
    Creats a video with Lanes that are detected from imput File to output File.
    Needs "output" Folder to store Video


    :param input_path: path to input video file
    :return: path to output video file
    :rtype: str
    '''

    output_path = "output/" + input_path.split("/")[-1]

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        #Algorythem for lane Detection:
        canny_frame = iF.canny(frame)

        cropped_frame = rD.region_of_interest(canny_frame)
        lines = __lines(cropped_frame)

        channel_canny_frame = cv2.cvtColor(canny_frame, cv2.COLOR_GRAY2BGR)


        processed = channel_canny_frame
        out.write(processed)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return output_path


def __lines(cropped_frame):
    '''
    sucht nach linien im gegebenen frame mittels Hough Transformation

    :param cropped_frame: frame in dem linien gesucht werden sollen(nur bereich im roi)
    :return: gefundene linien
    '''
    return cv2.HoughLinesP(
            cropped_frame,
            2,              #Schritte in denen gesucht wird (je kleiner desto genauer)
            np.pi / 180,    #Grad in denen gesucht wird (je kleiner desto genauer)
            100,            #Min anzahl für "votes" damits line wird (je größer desto weniger linien)
            np.array([]),
            minLineLength=40,   #Min abstand zwischen linien
            maxLineGap=5,       #Max abstand zwischen linien um zu verbinden
        )

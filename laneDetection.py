from pathlib import Path

import cv2
import numpy as np

import imageFilter as iF
import progressbar
import roiDetection as rD


def lane_detection(input_path):
    '''
    Creats a video with Lanes that are detected from imput File to output File.
    Needs "output" Folder to store Video


    :param input_path: path to input video file
    :return: path to output video file
    :rtype: str
    '''

    output_path = Path("output") / Path(input_path).name

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    i = 0
    max_index = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        #Algorythem for lane Detection:
        roi_frame = rD.region_of_interest(frame)
        bilateral_frame = iF.bilateral_filter(roi_frame)
        white_frame = iF.white_image(bilateral_frame)
        morph_frame = iF.morphological_cleanup(white_frame)
        cleanup = iF.clean_small_objects(morph_frame)
        canny_frame = iF.canny(cleanup)

        lines = __lines(canny_frame)
        filtered_lines = __throwaway_lines(frame, lines)
        #avrage_lines = __avrage_lines(frame, lines)
        lines_frame = __lines_frame(frame, filtered_lines)

        #processed = iF.image_overlay(frame, lines_frame)
        processed = iF.image_overlay(frame, lines_frame)
        out.write(processed)

        #Progress Bar
        progressbar.print_progress_bar(i, max_index, bar_length=40)
        i += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return output_path

def __throwaway_lines(frame, lines):
    '''
    wirft linien weg die außerhalb eines bestimmten winkel bereichs liegen

    :param frame: frame in dem die linien sind
    :param lines: linien die gefiltert werden sollen
    :return: gefilterte linien als array
    '''
    filtered_lines = []
    if lines is None:
        return np.array([])

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)

        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        slope = dy / dx
        angle = np.arctan(slope) * 180 / np.pi
        if 20 < abs(angle) < 60:  # noqa: PLR2004 ------------- nur linien mit einem winkel zwischen 20 und 60 grad behalten
            filtered_lines.append(line)
    return np.array(filtered_lines)


def __lines(cropped_frame):
    '''
    sucht nach linien im gegebenen frame mittels Hough Transformation

    :param cropped_frame: frame in dem linien gesucht werden sollen(nur bereich im roi)
    :return: gefundene linien as array
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

def __avrage_lines(frame, lines):
    '''
    berechnet aus den gegebenen linien die durchschnitts linien für links und rechts bezogen auf die steigung

    :param frame: frame in dem die linien sind
    :param lines: linien die gemittelt werden sollen
    :return: durchschnitts linien als array
    '''
    left_line = None
    right_line = None
    if lines is None:
        return np.array([])

    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    if len(left_fit) > 0:
        left_fit_avrage = np.average(left_fit, axis=0)
        left_line = __make_coordinates(frame, left_fit_avrage)
    if len(right_fit) > 0:
        right_fit_avrage = np.average(right_fit, axis=0)
        right_line = __make_coordinates(frame, right_fit_avrage)
    return np.array([line for line in [left_line, right_line] if line is not None])


def __make_coordinates(frame, line_parameters):
    slope, intercept = line_parameters
    y1 = frame.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def __lines_frame(frame, lines):
    '''
    fügt die gegebenen linien zu einem frame zusammen.
    dabei ist der frame so groß wie der originale frame

    :param frame: frame der größe des originalen frames
    :param lines: linien die zusammengefügt werden sollen
    :return: alle linien in einem frame
    '''
    line_frame = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_frame, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_frame

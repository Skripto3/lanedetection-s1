import sys
from pathlib import Path

import cv2
import error as e
import numpy as np

import imageFilter as iF
import progressbar
import roiDetection as rD


def main():
    print("Video processing started...")
    MIN_ARGS = 2  # noqa: N806
    if len(sys.argv) < MIN_ARGS:
        e.input_error("not enough arguments provided")
        sys.exit(1)
    elif len(sys.argv) == MIN_ARGS:
        sys.argv.append(None)

    cap = lane_detection(sys.argv[1], sys.argv[2])
    print("Video processing completed.")
    print("Processed video saved at:", cap)


def lane_detection(input_path, output_path):
    '''
    Creats a video with Lanes that are detected from imput File to output File.
    Detects lanes by applying various image processing techniques and Hough Transform.
    Output is saved in the specified output path or in an "output" folder if no output path is provided.


    :param input_path: path to input video file
    :return: path to output video file
    :rtype: str
    '''

    if output_path is None:
        output_path = Path("output") / Path(input_path).name
    else:
        output_path = Path(output_path) / Path(input_path).name

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
        lines_frame = __lines_frame(frame, filtered_lines)

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

if __name__ == "__main__":
    main()

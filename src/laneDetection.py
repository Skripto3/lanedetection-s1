import sys
from pathlib import Path

import cv2
import numpy as np

import error as e
import imageFilter as iF
import progressbar
import roiDetection as rD


def main():
    MIN_ARGS = 2  # noqa: N806
    if len(sys.argv) < MIN_ARGS:
        e.input_error("not enough arguments provided (minimum input_video_path)\n")
        sys.exit(1)
    elif len(sys.argv) == MIN_ARGS:
        sys.argv.append(None)

    print("Video processing started...")
    cap = lane_detection(sys.argv[1], sys.argv[2])
    print("Video processing completed.")
    print("Saved at: ", cap)


def lane_detection(input_path, output_path):
    '''
    Liest ein Video ein und verarbeitet jedes Frame um Fahrspuren zu erkennen.
    Das Ergebnis wird in einer neuen Videodatei gespeichert.
    Gibt daraufhin den Pfad der Ausgabedatei zurück.

    :param output_path: pfad zur Ausgabe Videodatei
    :param input_path: pfad zur Eingabe Videodatei
    :return: path zur Ausgabe Videodatei
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

    i = 1
    max_index = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        #Algorithmus für lane Detection:
        roi_frame = rD.region_of_interest(frame)
        bilateral_frame = iF.bilateral_filter(roi_frame)
        white_frame = iF.white_image(bilateral_frame)
        morph_frame = iF.morphological_cleanup(white_frame)
        cleanup = iF.clean_small_objects(morph_frame)
        canny_frame = iF.canny(cleanup)

        lines = __lines(canny_frame)
        filtered_lines = __throwaway_lines(lines)
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

def __throwaway_lines(lines):
    '''
    Wirft Linien weg die außerhalb eines bestimmten Winkelbereichs liegen

    :param lines: Linien die gefiltert werden sollen
    :return: gefilterte Linien als Array
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
        if 20 < abs(angle) < 60:  # noqa: PLR2004 ------------- nur linien mit einem Winkel zwischen 20 und 60 grad behalten
            filtered_lines.append(line)
    return np.array(filtered_lines)


def __lines(cropped_frame):
    '''
    Sucht nach Linien im gegebenen frame mittels Hough Transformation

    :param cropped_frame: frame in dem Linien gesucht werden sollen(nur bereich im roi)
    :return: gefundene Linien als Array
    '''
    return cv2.HoughLinesP(
            cropped_frame,
            2,              #Schritte in denen gesucht wird (je kleiner desto genauer/langsamer)
            np.pi / 180,    #Grad in denen gesucht wird (je kleiner desto genauer/langsamer)
            100,            #Min anzahl für "votes" damits line wird (je größer desto weniger Linien)
            np.array([]),
            minLineLength=40,   #Min länge für Linien
            maxLineGap=5,       #Max abstand zwischen Linien um zu verbinden
        )

def __lines_frame(frame, lines):
    '''
    fügt die gegebenen Linien mit einem frame zusammen.
    dabei ist der frame so groß wie der originale frame

    :param frame: frame der größe des originalen frames
    :param lines: Linien die zusammengefügt werden sollen
    :return: alle Linien in einem frame
    '''
    line_frame = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_frame, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_frame

if __name__ == "__main__":
    main()

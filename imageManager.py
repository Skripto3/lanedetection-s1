import sys
import time

import cv2

import imageFilter as iF
import roiDetection as rD


def play_video(path):
    cap = cv2.VideoCapture(path)

    i = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


        #avrage time per frame for each function
        time_spent ={
            "roi_detection": 0,
            "bilateral_filter": 0,
            "white_image": 0,
            "morphological_cleanup": 0,
            "clean_small_objects": 0,
            "canny": 0
        }
        start_time = time.time()

        #blur_frame = cv2.GaussianBlur(frame, (5, 5), 0)
        roi_frame = rD.region_of_interest(frame); time_spent["roi_detection"] = calc_time_in_ms(start_time, time.time()); start_time = time.time()
        bilateral_frame = iF.bilateral_filter(roi_frame) ; time_spent["bilateral_filter"] = calc_time_in_ms(start_time, time.time()); start_time = time.time()
        white_frame = iF.white_image(bilateral_frame) ; time_spent["white_image"] = calc_time_in_ms(start_time, time.time()); start_time = time.time()
        morph_frame = iF.morphological_cleanup(white_frame) ; time_spent["morphological_cleanup"] = calc_time_in_ms(start_time, time.time()); start_time = time.time()
        cleanup = iF.clean_small_objects(morph_frame) ; time_spent["clean_small_objects"] = calc_time_in_ms(start_time, time.time()); start_time = time.time()
        canny_frame = iF.canny(white_frame) ; time_spent["canny"] = calc_time_in_ms(start_time, time.time()); start_time = time.time()
        i += 1
        resize_frame = resize(cleanup, (800, 400))
        show_avrage_time(time_spent, i)
        cv2.imshow("video", resize_frame)
        if cv2.waitKey(1) == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

def split_play_video(path1, path2):
    output_path = "output/" + path1.split("/")[-1] + "_and_" + path2.split("/")[-1]

    cap = cv2.VideoCapture(path1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (1600, 800))

    cap1 = cv2.VideoCapture(path1)
    cap2 = cv2.VideoCapture(path2)
    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break
        height, width, _ = frame1.shape
        frame2_resized = cv2.resize(frame2, (width, height))
        combined_frame = cv2.hconcat([frame1, frame2_resized])
        out.write(resize(combined_frame, (1600, 800)))

        cv2.imshow("video", resize(combined_frame, (1600, 800)))
        if cv2.waitKey(1) == ord("q"):
            break
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

def resize(image, size):
    return cv2.resize(image, size)

def display_image(image):
    cv2.imshow('result', image)
    cv2.waitKey(0)

def calc_time_in_ms(start_time, end_time):
    if start_time == end_time:
        return 0
    return (end_time - start_time) * 1000

def show_avrage_time(time_spent, frame_count):
    stats = " | ".join(
        f"{k}: {v/frame_count:.2f}ms"
        for k, v in time_spent.items()
    )
    sys.stdout.write("\r" + stats + " " * 10)
    sys.stdout.flush()

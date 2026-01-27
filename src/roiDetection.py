#Dummy classe für roiDetection.py
#Eig. Ziel: Region of Interest (ROI) im Bild definieren...

def region_of_interest(frame):
    #obere hälfte des bildes schwarzen
    roi_frame = frame.copy()
    roi_frame[:int(roi_frame.shape[0] / 3), :] = 0

    return roi_frame

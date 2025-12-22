import sys

import laneDetection as lD

print("Video processing statrted...")
cap = lD.lane_detection(sys.argv[1])
print("Video processing completed.")
print("Processed video saved at:", cap)
#cap = lD.lane_detection("/home/justus/Dokumente/Programmierne/Git/lanedetection-s1/data/test4.mp4")

#iM.play_video(cap)

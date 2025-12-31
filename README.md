# lanedetection-s1
A lane detection project using computer vision techniques.
Writen as a projact for class.

---

## Features
- Noise filtering
- Canny edge detection
- Hough Transform lane detection
- Supports video input and video output
- CLI command for easy usage

---

## Requirements
- Python 3.0+
- Libraries:
  - numpy
  - opencv-python (cv2)
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  
---

## Usage
Run in terminal:
(If output path not specefied it is placed in the output folder)

```bash
lanedetection <input_video_path> [<output_video_path>]

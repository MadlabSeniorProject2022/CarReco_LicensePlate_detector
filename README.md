# Lean Detector

** Original source code for yolov7 from https://github.com/WongKinYiu/yolov7

Implement detect function for yolo model that use in web sevice.
- Can detect image of video that source is static file only.
- Reduce time to reload same model when get new request.
- Implement structure easier to call and handle result after detected.

To use it sample is in demo.ipynb, download .pt model into main directory before use lean_detect.py

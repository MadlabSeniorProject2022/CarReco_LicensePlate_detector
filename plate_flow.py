from lean_detect import UseModel
import cv2
import os
import time

from deskew import determine_skew
import numpy as np
from typing import Tuple, Union
import math

import pytesseract

plate_model = UseModel("LPD150.pt")

def scale_up(file_path):
    if not os.path.exists("./up_scale"):
        os.mkdir("/up_scale")
    sr = cv2.dnn_superres.DnnSuperResImpl_create()

    modelPath = 'upscaling\LapSRN_x8.pb'

    img = cv2.imread(file_path)

    sr.readModel(modelPath)
    sr.setModel("lapsrn", int(512/img.size[2]))

    
    img = sr.upsample(img)
    dst_path = f"up_scale/{time.time()*100000}.jpg"
    
    cv2.imwrite(dst_path, img)

    return dst_path

def process(cv_image: cv2.Mat):
    plate_image = cv2.convertScaleAbs(cv_image, alpha=(255.0))
    # convert to grayscale and blur the image
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)

    # Applied inversed thresh_binary 
    binary = cv2.threshold(blur, 80, 200,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C)[1]

    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
    return thresh_mor

def rotate(image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

def fix_rotate(cv_image: cv2.Mat):
    angle = determine_skew(cv_image)
    rotated = rotate(cv_image, angle, (0, 0, 0))
    return rotated

def get_word(file_path):
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    thresh_mor = process(img)
    rotated = fix_rotate(thresh_mor)
    extractedInformation = pytesseract.image_to_string(rotated)
    extractedInformation = extractedInformation.strip()
    return extractedInformation[-4:]

def full_flow(file_path):
    scale_path = scale_up(file_path)
    return get_word(scale_path)


import os

import torch
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

random.seed(1)

class UseModel:
    def __init__(self, weights, device = "", confident = 0.25, iou = 0.45, detect_class = None, normalize_size = 640, is_trace = True) -> None:
        set_logging()
        self.device = select_device(device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.norm_size = normalize_size
        self.detect_class = detect_class
        self.confident = confident
        self.iou = iou
        # Load model
        model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.stride = int(model.stride.max())  # model stride
        self.norm_size= check_img_size(self.norm_size, s=self.stride)  # check img_size

        if is_trace:
            model = TracedModel(model, self.device, self.norm_size)

        if self.half:
            model.half()  # to FP16
        
        if self.device.type != 'cpu':
            model(torch.zeros(1, 3, self.norm_size, self.norm_size).to(self.device).type_as(
                next(model.parameters())))  # run once
        
        self.model = model

    def detect (self, source, do_function = None):
        old_img_w = old_img_h = self.norm_size
        old_img_b = 1
        dataset = LoadImages(source, img_size=self.norm_size, stride=self.stride)
        for path, img, im0s, vid_cap in dataset:
            print(path)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if self.device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    self.model(img, augment=False)[0]

            pred = self.model(img, augment=False)[0]    
            pred = non_max_suppression(pred, self.confident, self.iou, classes=self.detect_class, agnostic=False)

            # Do function that include
            #if do_function != None:
            #    return do_function(pred, im0s, dataset)
            # If function not include, return as predicted value
            #return pred

            for i, det in enumerate(pred):
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                # normalization gain whwh
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if do_function != None:
                            do_function(xyxy, im0, conf, cls)
        
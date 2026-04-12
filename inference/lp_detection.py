import numpy as np
from .base_onnx import BaseONNX
from utils.utils import DetectedObject
from utils.yolo_utils import process_yolo_output

class LPDetection(BaseONNX):
    def __init__(self):
        self.model_path = "./models/lp_detection-yolov4_1_3_416_416_static-simp.onnx"
        super().__init__(self.model_path, (416, 416))
        self.classes = ["antiga", "vermelha", "nova"]
        self.conf_thresh = 0.5

    def run_inference(self, image: np.ndarray) -> list[DetectedObject]:
        input_data = self.get_biddings(image)
        outputs = self.session.run(None, {self.input_name: input_data})
        h, w = image.shape[:2]
        return process_yolo_output(outputs, h, w, self.conf_thresh, self.classes)

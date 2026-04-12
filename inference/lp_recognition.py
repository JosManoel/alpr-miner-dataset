import numpy as np
from .base_onnx import BaseONNX
from utils.utils import DetectedObject
from utils.yolo_utils import process_yolo_output

class LPRecognition(BaseONNX):
    def __init__(self):
        self.model_path = "./models/lp_recognition-yolov4_1_3_64_224_static-simp.onnx"
        super().__init__(self.model_path, (64, 224))
        self.classes = ["segmento"]
        self.conf_thresh = 0.3

    def run_inference(self, image: np.ndarray) -> list[DetectedObject]:
        input_data = self.get_biddings(image)
        outputs = self.session.run(None, {self.input_name: input_data})
        h, w = image.shape[:2]
        return process_yolo_output(outputs, h, w, self.conf_thresh, self.classes)

import cv2
import numpy as np
import onnxruntime as ort
from typing import Any
from abc import ABC, abstractmethod


class BaseONNX:
    def __init__(self, model_path: str, input_shape: tuple[int, int]):
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = input_shape

    def get_biddings(self, image: np.ndarray) -> np.ndarray:
        h, w = self.input_shape
        img_resized = cv2.resize(image, (w, h))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_t = np.transpose(img_norm, (2, 0, 1))
        img_batch = np.expand_dims(img_t, axis=0)
        return img_batch

    @abstractmethod
    def run_inference(self, image: np.ndarray) -> Any:
        pass

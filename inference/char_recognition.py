import cv2
import numpy as np
import onnxruntime as ort
from utils.utils import DetectedObject, BBoxModel

class CharacterRecognition:
    def __init__(self):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        self.path_letra_antiga = "./models/recognition_letra_antiga.onnx"
        self.path_letra_nova = "./models/recognition_letra_nova.onnx"
        self.path_num_antiga = "./models/recognition_numero_antiga.onnx"
        self.path_num_nova = "./models/recognition_numero_nova.onnx"

        self.sess_letra_antiga = ort.InferenceSession(self.path_letra_antiga, providers=providers)
        self.sess_letra_nova = ort.InferenceSession(self.path_letra_nova, providers=providers)
        self.sess_num_antiga = ort.InferenceSession(self.path_num_antiga, providers=providers)
        self.sess_num_nova = ort.InferenceSession(self.path_num_nova, providers=providers)

        self.input_shape = (20, 20)
        self.letter_classes = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        self.number_classes = list('0123456789')

        self.old_lp_number_pos = [3, 4, 5, 6]
        self.new_lp_number_pos = [3, 5, 6]

    def get_biddings(self, image: np.ndarray, should_invert: bool = False) -> np.ndarray:
        # Redimensiona para o tamanho esperado pela rede (20, 20)
        img_resized = cv2.resize(image, self.input_shape)

        # Converte para escala de cinza (1 canal)
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        if should_invert:
            img_gray = np.invert(img_gray)

        # O shape atual é (20, 20).
        # Adicionamos uma dimensão no final para representar o canal: fica (20, 20, 1)
        img_expanded = np.expand_dims(img_gray, axis=-1)

        # Adicionamos uma dimensão no inicio para representar o batch: fica (1, 20, 20, 1)
        img_batch = np.expand_dims(img_expanded, axis=0)

        # Normaliza os valores dos pixels para o intervalo [0, 1]
        img_batch = img_batch.astype(np.float32) / 255.0

        return img_batch

    def run_inference(self, image: np.ndarray, plate_type: str, char_index: int) -> DetectedObject:
        should_invert = (plate_type == "vermelha")
        input_data = self.get_biddings(image, should_invert)

        is_old = (plate_type in ["antiga", "vermelha"])

        if is_old:
            is_number = char_index in self.old_lp_number_pos
            session = self.sess_num_antiga if is_number else self.sess_letra_antiga
            classes = self.number_classes if is_number else self.letter_classes
        else:
            is_number = char_index in self.new_lp_number_pos
            session = self.sess_num_nova if is_number else self.sess_letra_nova
            classes = self.number_classes if is_number else self.letter_classes

        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: input_data})

        predictions = output[0][0]
        max_idx = int(np.argmax(predictions))
        confidence = float(predictions[max_idx])
        label = classes[max_idx]

        h, w = image.shape[:2]
        dummy_bbox = BBoxModel(x=0, y=0, w=w, h=h)

        return DetectedObject(bbox=dummy_bbox, label=label, confidence=confidence)

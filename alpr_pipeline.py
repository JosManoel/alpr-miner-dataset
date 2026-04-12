import uuid
import numpy as np
from inference import CarDetection, LPDetection, LPRecognition, CharacterRecognition
from utils import DatasetManager
from utils import crop_image

class ALPRPipeline:
    def __init__(self, dataset_manager: DatasetManager,
                 min_conf_car: float = 0.7, min_conf_plate: float = 0.7, min_conf_ocr: float = 0.7):

        self.car_det = CarDetection()
        self.lp_det = LPDetection()
        self.lp_seg = LPRecognition()
        self.char_rec = CharacterRecognition()

        self.car_det.conf_thresh = min_conf_car
        self.lp_det.conf_thresh = min_conf_plate
        self.lp_seg.conf_thresh = min_conf_ocr
        self.min_conf_ocr = min_conf_ocr

        self.db = dataset_manager

    def process_frame(self, frame: np.ndarray, source_video: str, frame_num: int):
        # 1. Detecção de Veículos
        cars = self.car_det.run_inference(frame)

        for car in cars:
            if car.confidence < self.car_det.conf_thresh:
                continue

            car_id = f"car_{uuid.uuid4().hex[:8]}"
            self.db.save_record("car", car_id, str(frame_num), frame, car.bbox.xyxy, car.label, car.confidence, source_video, (1280, 720))

            car_crop = crop_image(frame, car.bbox)
            if car_crop.size == 0:
                continue

            # 2. Detecção de Placas
            plates = self.lp_det.run_inference(car_crop)
            for plate in plates:
                if plate.confidence < self.lp_det.conf_thresh:
                    continue

                plate_id = f"plt_{uuid.uuid4().hex[:8]}"
                self.db.save_record("plate", plate_id, car_id, car_crop, plate.bbox.xyxy, plate.label, plate.confidence, source_video, (640, 640))

                plate_crop = crop_image(car_crop, plate.bbox)
                if plate_crop.size == 0:
                    continue

                # 3. Segmentação de Caracteres
                char_segments = self.lp_seg.run_inference(plate_crop)
                char_segments.sort(key=lambda c: c.bbox.x)

                valid_chars = []
                for char_seg in char_segments:
                    if char_seg.confidence >= self.lp_seg.conf_thresh:
                        valid_chars.append(char_seg)

                if len(valid_chars) != 7:
                    continue

                full_text = ""
                confidences = []
                ocr_group_id = f"ocr_{uuid.uuid4().hex[:8]}"

                # 4. Reconhecimento de Caracteres (OCR)
                for idx, char_seg in enumerate(valid_chars):
                    char_crop = crop_image(plate_crop, char_seg.bbox)
                    if char_crop.size == 0:
                        continue

                    recognized = self.char_rec.run_inference(char_crop, plate.label, idx)
                    if recognized.confidence < self.min_conf_ocr:
                        break

                    full_text += recognized.label
                    confidences.append(recognized.confidence)

                    char_id = f"{ocr_group_id}_{idx}"
                    # id_font referencia o plate_id (UUID)
                    self.db.save_record("char", char_id, plate_id, plate_crop, char_seg.bbox.xyxy, recognized.label, recognized.confidence, source_video, (224, 64))

                # 5. Consolidação do OCR se tiver 7 caracteres
                if len(full_text) == 7:
                    avg_conf = sum(confidences) / len(confidences)
                    # id_font referencia o plate_id (UUID)
                    self.db.save_record("ocr", ocr_group_id, plate_id, None, plate.bbox.xyxy, full_text, avg_conf, source_video, None)

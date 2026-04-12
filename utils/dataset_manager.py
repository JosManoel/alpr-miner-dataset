import os
import cv2
import pandas as pd
import numpy as np

class DatasetManager:
    def __init__(self, base_dir: str = "dataset_output"):
        self.base_dir = base_dir
        self.car_dir = os.path.join(base_dir, "cars")
        self.plate_dir = os.path.join(base_dir, "plates")
        self.char_dir = os.path.join(base_dir, "chars")
        self.ocr_dir = os.path.join(base_dir, "ocrs")

        for d in [self.car_dir, self.plate_dir, self.char_dir, self.ocr_dir]:
            os.makedirs(d, exist_ok=True)

        self.car_records = []
        self.plate_records = []
        self.char_records = []
        self.ocr_records = []

    def save_record(self, dataset_type: str, record_id: str, id_font: str, image: np.ndarray | None,
                    bbox: tuple, label: str, conf: float, source_video: str, resize_dim: tuple | None):

        img_resized = None
        if image is not None and resize_dim is not None:
            img_resized = cv2.resize(image, resize_dim)

        if dataset_type == "car":
            img_path = os.path.join(self.car_dir, f"{record_id}.jpg")
            self.car_records.append([record_id, id_font, str(bbox), label, conf, source_video])
            if img_resized is not None:
                cv2.imwrite(img_path, img_resized)

        elif dataset_type == "plate":
            img_path = os.path.join(self.plate_dir, f"{record_id}.jpg")
            self.plate_records.append([record_id, id_font, str(bbox), label, conf, source_video])
            if img_resized is not None:
                cv2.imwrite(img_path, img_resized)

        elif dataset_type == "char":
            img_path = os.path.join(self.char_dir, f"{record_id}.jpg")
            self.char_records.append([record_id, id_font, str(bbox), label, conf, source_video])
            if img_resized is not None:
                cv2.imwrite(img_path, img_resized)

        elif dataset_type == "ocr":
            img_path = os.path.join(self.ocr_dir, f"{record_id}.jpg")
            self.ocr_records.append([record_id, id_font, str(bbox), label, conf, source_video])
            if img_resized is not None:
                cv2.imwrite(img_path, img_resized)

    def export_csvs(self):
        columns = ["id", "id_font", "bbox", "label", "conf", "source_video"]

        # cria as tabelas
        pd.DataFrame(self.car_records, columns=columns).to_csv(os.path.join(self.base_dir, "cars_dataset.csv"), index=False)
        pd.DataFrame(self.plate_records, columns=columns).to_csv(os.path.join(self.base_dir, "plates_dataset.csv"), index=False)
        pd.DataFrame(self.char_records, columns=columns).to_csv(os.path.join(self.base_dir, "chars_dataset.csv"), index=False)
        pd.DataFrame(self.ocr_records, columns=columns).to_csv(os.path.join(self.base_dir, "ocrs_dataset.csv"), index=False)

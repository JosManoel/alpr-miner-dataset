import os
import cv2
import pandas as pd
import numpy as np

class DatasetManager:
    def __init__(self, base_dir: str = "dataset_output"):
        #salva no /dados
        self.base_dir = "/dados/canela/alpr-dataset/output_dataset"

        self.car_dir = os.path.join(self.base_dir, "cars")
        self.plate_dir = os.path.join(self.base_dir, "plates")
        self.char_dir = os.path.join(self.base_dir, "chars")
        self.ocr_dir = os.path.join(self.base_dir, "ocrs")

        for d in [self.car_dir, self.plate_dir, self.char_dir, self.ocr_dir]:
            os.makedirs(d, exist_ok=True)

        self.columns = ["id", "id_font", "bbox", "label", "conf", "source_video"]


        # Mapeamento dos CSVs
        self.csv_paths = {
            "car": os.path.join(self.base_dir, "cars_dataset.csv"),
            "plate": os.path.join(self.base_dir, "plates_dataset.csv"),
            "char": os.path.join(self.base_dir, "chars_dataset.csv"),
            "ocr": os.path.join(self.base_dir, "ocrs_dataset.csv")
        }

        # Cria os arquivos CSV
        for path in self.csv_paths.values():
            if not os.path.exists(path):
                pd.DataFrame(columns=self.columns).to_csv(path, index=False)

    def save_record(self, dataset_type: str, record_id: str, id_font: str, image: np.ndarray | None,
                    bbox: tuple, label: str, conf: float, source_video: str, resize_dim: tuple | None):

        img_resized = None
        if image is not None and resize_dim is not None:
            img_resized = cv2.resize(image, resize_dim)

        # escolhe as pastas
        if dataset_type == "car":
            img_dir = self.car_dir
        elif dataset_type == "plate":
            img_dir = self.plate_dir
        elif dataset_type == "char":
            img_dir = self.char_dir
        elif dataset_type == "ocr":
            img_dir = self.ocr_dir
        else:
            return

        # Salva a imagem
        if img_resized is not None:
            img_path = os.path.join(img_dir, f"{record_id}.jpg")
            cv2.imwrite(img_path, img_resized)

        # Salva linha no csv
        row_data = [[record_id, id_font, str(bbox), label, conf, source_video]]
        csv_path = self.csv_paths[dataset_type]

        pd.DataFrame(row_data, columns=self.columns).to_csv(csv_path, mode='a', header=False, index=False)

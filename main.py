import cv2
import os
from tqdm import tqdm
from utils import DatasetManager
from alpr_pipeline import ALPRPipeline

def main():
    videos = [
        "./data/natal.webm",
        "./data/pelas_ruas_salvador.mp4",
        "./data/dirigir_em_salvador.mp4",
        "./data/salvador_bahia.mp4",
        "./data/sao_paulo-tiete.webm",
        "./data/1h20min-sao_paulo.mp4",
        "./data/rio_De_janeiro.mp4",
        "./data/rio_de_janeiro_asmr.mp4",
        "./data/rio_de_janeiro_castelo_branco.mp4"
    ]

    dataset_manager = DatasetManager("output_dataset")

    pipeline = ALPRPipeline(
        dataset_manager=dataset_manager,
        min_conf_car=0.9,
        min_conf_plate=0.9,
        min_conf_ocr=0.9
    )

    for video_path in videos:
        print(f"Processando vídeo: {video_path}")
        if not os.path.exists(video_path):
            print(f"Vídeo não encontrado: {video_path}")
            continue

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_name = os.path.basename(video_path)
        frame_num = 0

        with tqdm(total=total_frames, desc=video_name) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_num += 1
                pipeline.process_frame(frame, video_name, frame_num)
                pbar.update(1)

        cap.release()

    print("Processamento concluído. Datasets exportados.")

if __name__ == "__main__":
    main()

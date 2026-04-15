from typing import Any


from types import NotImplementedType


import pandas as pd
import os
import shutil


base_dir = "/dados/canela/alpr-dataset/output_dataset"
out_dir = "./dataset-final"

df_cars = pd.read_csv(os.path.join(base_dir, "cars_dataset.csv"))
df_plates = pd.read_csv(os.path.join(base_dir, "plates_dataset.csv"))
df_ocrs = pd.read_csv(os.path.join(base_dir, "ocrs_dataset.csv"))

os.makedirs(os.path.join(out_dir, "cars"), exist_ok=True)
os.makedirs(os.path.join(out_dir, "plates"), exist_ok=True)
os.makedirs(os.path.join(out_dir, "ocrs"), exist_ok=True)

# Filtros
df_conf = df_ocrs[(df_ocrs['conf'] >= 0.9) & (df_ocrs['conf'] <= 1.0)].copy() # onf entre 0.9 e 1.0

_aux_count = df_conf['label'].value_counts()
labels_finais = _aux_count[(_aux_count >= 10) & (_aux_count <= 20000)].index # min de 10 ocorrencias

temp_df = df_conf[df_conf['label'].isin(labels_finais)].copy()


temp_df['rn'] = temp_df.groupby('label').cumcount()
temp_df['total'] = temp_df.groupby('label')['label'].transform('count')

# Cria as regras para pegar a 1ª, a do meio e a última
first_plate: Any | NotImplementedType = temp_df['rn'] == 0
med_plate = temp_df['rn'] == (temp_df['total'] // 2)
last_plate = temp_df['rn'] == (temp_df['total'] - 1)

# Filtro tabela de ocr
df_ocrs_curado = temp_df[first_plate | med_plate | last_plate].copy()
df_ocrs_curado.drop(columns=['rn', 'total'], inplace=True)
df_ocrs_curado.reset_index(drop=True, inplace=True)

# Filtro tabela de placas
plate_ids = df_ocrs_curado['id_font'].unique()
df_plates_curado = df_plates[df_plates['id'].isin(plate_ids)].copy()

# Filtro tabelca de carros
car_ids = df_plates_curado['id_font'].unique()
df_cars_curado = df_cars[df_cars['id'].isin(car_ids)].copy()


ocrs_renamed = df_ocrs_curado.rename(columns={
    'id': 'ocr_id', 'id_font': 'plate_id', 'bbox': 'bbox_ocr',
    'label': 'ocr_texto', 'conf': 'ocr_conf'
})[['ocr_id', 'plate_id', 'bbox_ocr', 'ocr_texto', 'ocr_conf', 'source_video']]

plates_renamed = df_plates_curado.rename(columns={
    'id': 'plate_id', 'id_font': 'car_id', 'bbox': 'bbox_plate',
    'label': 'tipo_placa', 'conf': 'plate_conf'
})[['plate_id', 'car_id', 'bbox_plate', 'tipo_placa', 'plate_conf']]

cars_renamed = df_cars_curado.rename(columns={
    'id': 'car_id', 'id_font': 'frame_num', 'bbox': 'bbox_car',
    'conf': 'car_conf'
})[['car_id', 'frame_num', 'bbox_car', 'car_conf']]

# Tablema com todos os dados
wide_df = ocrs_renamed.merge(plates_renamed, on='plate_id', how='left')
wide_df = wide_df.merge(cars_renamed, on='car_id', how='left')

wide_df = wide_df[[
    'source_video', 'frame_num',
    'car_id', 'bbox_car', 'car_conf',
    'plate_id', 'bbox_plate', 'tipo_placa', 'plate_conf',
    'ocr_id', 'bbox_ocr', 'ocr_texto', 'ocr_conf'
]]


df_cars_curado.to_csv(os.path.join(out_dir, "cars_dataset.csv"), index=False)
df_plates_curado.to_csv(os.path.join(out_dir, "plates_dataset.csv"), index=False)
df_ocrs_curado.to_csv(os.path.join(out_dir, "ocrs_dataset.csv"), index=False)
wide_df.to_csv(os.path.join(out_dir, "wide_dataset.csv"), index=False)


# Copia imagens
def copiar_imagens(df, folder_name):
    for img_id in df['id']:
        src = os.path.join(base_dir, folder_name, f"{img_id}.jpg")
        dst = os.path.join(out_dir, folder_name, f"{img_id}.jpg")
        if os.path.exists(src):
            shutil.copy2(src, dst)

print("Copiando imagens de Carros...")
copiar_imagens(df_cars_curado, "cars")

print("Copiando imagens de Placas...")
copiar_imagens(df_plates_curado, "plates")

print("Copiando imagens de OCRs...")
copiar_imagens(df_ocrs_curado, "ocrs")


print(f"Sucesso! Dataset curado salvo em: {out_dir}")
print(f"Total de Imagens Carro extraídas: {len(df_cars_curado)}")


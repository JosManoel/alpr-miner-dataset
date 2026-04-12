import cv2
import numpy as np
from .utils import BBoxModel, DetectedObject

def process_yolo_output(outputs, img_h: int, img_w: int, conf_thresh: float, classes: list[str]) -> list[DetectedObject]:
    boxes_data = outputs[0]
    scores_data = outputs[1]

    boxes_data = boxes_data[:, :, 0]
    max_conf = np.max(scores_data, axis=2)
    max_id = np.argmax(scores_data, axis=2)

    detected_objects = []
    num_classes = len(classes)

    for i in range(boxes_data.shape[0]):
        mask = max_conf[i] > conf_thresh
        l_box_array = boxes_data[i, mask, :]
        l_max_conf = max_conf[i, mask]
        l_max_id = max_id[i, mask]

        for j in range(num_classes):
            cls_mask = l_max_id == j
            if not np.any(cls_mask):
                continue

            cls_boxes = l_box_array[cls_mask]
            cls_scores = l_max_conf[cls_mask]

            curr_boxes = []
            for box in cls_boxes:
                # corrcao para x1x2 -> [x1, y1, x2, y2]
                x1 = box[0] * img_w
                y1 = box[1] * img_h
                x2 = box[2] * img_w
                y2 = box[3] * img_h

                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)

                if w <= 0 or h <= 0:
                    w = int(box[2] * img_w)
                    h = int(box[3] * img_h)
                    x = int(box[0] * img_w - w / 2)
                    y = int(box[1] * img_h - h / 2)

                curr_boxes.append([x, y, w, h])

            indices = cv2.dnn.NMSBoxes(curr_boxes, cls_scores.tolist(), conf_thresh, 0.4)

            if len(indices) > 0:
                for idx in indices.flatten():
                    bbox = BBoxModel(
                        x=curr_boxes[idx][0],
                        y=curr_boxes[idx][1],
                        w=curr_boxes[idx][2],
                        h=curr_boxes[idx][3]
                    )
                    detected_objects.append(DetectedObject(
                        bbox=bbox,
                        label=classes[j],
                        confidence=float(cls_scores[idx])
                    ))

    return detected_objects

def crop_image(img: np.ndarray, bbox: BBoxModel) -> np.ndarray:
    x1, y1, x2, y2 = bbox.xyxy
    h_img, w_img = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    return img[y1:y2, x1:x2]

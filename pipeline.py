import os
import cv2
import zipfile
from ultralytics import YOLO

import torchvision
import torch

# Monkey-patch NMS to force CPU
_orig_nms = torchvision.ops.nms
def nms_cpu(boxes, scores, iou_threshold):
    return _orig_nms(boxes.cpu(), scores.cpu(), iou_threshold)
torchvision.ops.nms = nms_cpu


class VideoDetectionPipeline:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

        self.class_thresholds = {
            0: 0.7, 1: 0.65, 2: 0.65, 3: 0.67, 4: 0.5,
            5: 0.45, 6: 0.50, 7: 0.65, 8: 0.7,
            9: 0.60, 10: 0.55, 11: 0.65, 12: 0.50
        }

        self.class_names = {
            0: "bike", 1: "auto", 2: "truck", 3: "tempo",
            4: "traffic sign", 5: "traffic light", 6: "bicycle",
            7: "bus", 8: "car", 9: "person", 10: "animal",
            11: "rider", 12: "caravan"
        }

        self.class_colors = {
            0: (0, 255, 255),     1: (0, 165, 255),    2: (0, 0, 255),
            3: (255, 0, 0),       4: (255, 255, 0),    5: (128, 0, 128),
            6: (0, 255, 0),       7: (255, 0, 255),    8: (0,100,0),
            9: (0, 128, 255),     10: (42, 42, 165),   11: (203, 192, 255),
            12: (192, 192, 192)
        }

    def run(self, video_path, output_base_dir, progress_callback=None):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        raw_dir = os.path.join(output_base_dir, video_name, "raw")
        annotated_dir = os.path.join(output_base_dir, video_name, "annotated")
        labels_dir = os.path.join(output_base_dir, video_name, "labels")

        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(annotated_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Failed to open {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = int(fps * 3) if fps > 0 else 90
        idx = 0

        while idx < frame_count:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: break

            raw_path = os.path.join(raw_dir, f"{idx:05d}.jpg")
            cv2.imwrite(raw_path, frame)

            result = self.model(frame)[0]
            boxes = result.boxes
            filtered = []

            for box, cls, conf in zip(boxes.xywhn, boxes.cls, boxes.conf):
                cid = int(cls.item())
                if conf.item() >= self.class_thresholds.get(cid, 0.5):
                    filtered.append((box, cls, conf))

            for box, cls, conf in filtered:
                x, y, w, h = box
                cid = int(cls.item())
                name = self.class_names.get(cid, str(cid))
                color = self.class_colors.get(cid, (0, 255, 0))
                h_, w_ = frame.shape[:2]
                x1 = int((x - w/2) * w_)
                y1 = int((y - h/2) * h_)
                x2 = int((x + w/2) * w_)
                y2 = int((y + h/2) * h_)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            anno_path = os.path.join(annotated_dir, f"{idx:05d}.jpg")
            cv2.imwrite(anno_path, frame)

            label_path = os.path.join(labels_dir, f"{idx:05d}.txt")
            with open(label_path, "w") as f:
                for box, cls, _ in filtered:
                    x, y, w, h = box.tolist()
                    cid = int(cls.item())
                    f.write(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

            if progress_callback:
                progress_callback(min(idx / frame_count, 1.0))

            idx += interval

        cap.release()
        
        return os.path.join(output_base_dir, video_name)

        # zip_path = os.path.join(output_base_dir, f"{video_name}_results.zip")
        # with zipfile.ZipFile(zip_path, "w") as z:
        #     for root, _, files in os.walk(os.path.join(output_base_dir, video_name)):
        #         for file in files:
        #             full_path = os.path.join(root, file)
        #             arcname = os.path.relpath(full_path, output_base_dir)
        #             z.write(full_path, arcname)
        # return zip_path



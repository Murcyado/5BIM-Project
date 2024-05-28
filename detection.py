import cv2
import torch
import os

class ObjectDetection:
    def __init__(self, model_name='yolov5s'):
        # Charger le modèle YOLOv5
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        self.classes = self.model.names
        self.target_classes = ["laptop", "mouse"]  # Classes cibles: ordinateurs et souris
        self.output_dir = "detected_objects"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def detect_objects(self, frame):
        results = self.model(frame)
        detections = []
        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = det
            label = self.classes[int(cls)]
            if label in self.target_classes:
                detections.append((label, (int(x1), int(y1), int(x2), int(y2)), conf))
        return detections

    def save_detection(self, frame, label, bbox, frame_count):
        label_dir = os.path.join(self.output_dir, label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        x1, y1, x2, y2 = bbox
        obj_img = frame[y1:y2, x1:x2]
        if obj_img.size > 0:
            cv2.imwrite(os.path.join(label_dir, f"{label}_{frame_count}.jpg"), obj_img)

class CameraStream:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id)
        self.detector = ObjectDetection()
        self.frame_count = 0

    def process_stream(self):
        if not self.cap.isOpened():
            print(f"Erreur d'ouverture de la caméra {self.camera_id}")
            return

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Erreur dans la capture vidéo")
                break

            detections = self.detector.detect_objects(frame)
            counts = {}

            for label, bbox, conf in detections:
                counts[label] = counts.get(label, 0) + 1
                color = (0, 255, 0) if label == "mouse" else (255, 0, 0)
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                self.detector.save_detection(frame, label, bbox, self.frame_count)

            y_offset = 30
            for label, count in counts.items():
                cv2.putText(frame, f"{label}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                y_offset += 30

            cv2.imshow(f'Camera Stream {self.camera_id} - YOLOv5 Object Detection', frame)
            self.frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    stream = CameraStream(0)  # Utiliser la webcam de l'ordinateur
    stream.process_stream()

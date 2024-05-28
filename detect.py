import cv2
import torch
import os

class Detect_Objet:
    def __init__(self, model_name='yolov5s'):
        #Charger modèle YOLOv5
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        self.classes = self.model.names
        self.output_dir = "objets_detectes"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.target_classes = ['laptop', 'mouse']

    
    def objets_detectes(self,frame):
        resultat = self.model(frame)
        detections = []
        for detect in resultat.xyxy[0]:
            x1, x2, y1, y2, conf, cls = detect
            label = self.classes[int(cls)]
            if label in self.target_classes:
                detections.append((label, (int(x1), int(y1), int(x2), int(y2)), conf))
        return detections
    
    # Sauvegarde des images dans le dossier "objets_detectes"
    def sauvegarde_detection(self, frame, label, bbox, frame_count):
        label_dir = os.path.join(self.output_dir, label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        x1, y1, x2, y2 = bbox
        obj_img = frame[y1:y2, x1:x2]
        if obj_img.size > 0:
            cv2.imwrite(os.path.join(label_dir, f"{label}_{frame_count}.jpg"), obj_img)

class StreamWebcam: 
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id)
        self.detector = Detect_Objet()
        self.frame_count = 0

    def stream_processus(self):
        if not self.cap.isOpened():
            print(f"Erreur d'ouverture de la caméra {self.camera_id}")
            return
        
        while True:
            ret, frame = self.cap.read()
            if not ret: 
                print("Erreur de capture vidéo")
                break

            detections = self.detector.objets_detectes(frame)
            counts = {}

            for label, bbox, conf in detections:
                counts[label] = counts.get(label, 0) + 1
                color = (0, 255, 0) if label == "mouse" else (255, 0, 0)
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                self.detector.sauvegarde_detection(frame, label, bbox, self.frame_count)

            y_offset = 30
            for label, count in counts.items():
                cv2.putText(frame, f"{label}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                y_offset += 30

            cv2.imshow(f'Stream de la caméra {self.camera_id} - YOLOv5 Détection Objet', frame)
            self.frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    stream = StreamWebcam(0)
    stream.stream_processus()

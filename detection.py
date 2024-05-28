import cv2
import numpy as np
from dotenv import load_dotenv
import os
import aiohttp
import asyncio
import torch

load_dotenv()

# Charger le modèle YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Définir les classes de YOLOv5
CLASSES = model.names

transport_classes = {"aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train"}
living_classes = {"bird", "cat", "cow", "dog", "horse", "person", "sheep"}
object_classes = {"bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"}

# Créer un dossier pour stocker les images capturées
output_dir = "detected_objects"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

async def fetch_camera_data():
    api_key = os.getenv('API_KEY')
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    url = "https://api.netatmo.com/api/gethomesdata"

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                return data
            else:
                print(f"Erreur: {response.status}")
                print(await response.text())
                return None

async def display_camera_streams(camera_url):

    # Initialiser la capture vidéo pour la caméra réseau
    cap = cv2.VideoCapture(camera_url + "/live/files/low/index.m3u8")

    if not cap.isOpened():
        print("Erreur d'ouverture de la caméra réseau")
        return

    frame_count = 0
    while True:
        # Lire une image de la caméra réseau
        ret, frame = cap.read()

        if not ret:
            print("Erreur dans la capture vidéo")
            break

        # Utiliser YOLOv5 pour la détection
        results = model(frame)

        # Compter les occurrences des objets détectés
        counts = {}
        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = det
            label = CLASSES[int(cls)]
            counts[label] = counts.get(label, 0) + 1

            if label in transport_classes:
                color = (255, 0, 0)  # bleu
            elif label in living_classes:
                color = (0, 255, 0)  # vert
            elif label in object_classes:
                color = (0, 255, 255)  # jaune
            else:
                color = (255, 255, 255)  # blanc par défaut
            
            # Dessiner les boîtes et les étiquettes
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Créer un dossier pour chaque label s'il n'existe pas
            label_dir = os.path.join(output_dir, label)
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)

            # Enregistrer l'image de l'objet détecté
            obj_img = frame[int(y1):int(y2), int(x1):int(x2)]
            if obj_img.size > 0:  # Vérifier que l'image n'est pas vide
                cv2.imwrite(os.path.join(label_dir, f"{label}_{frame_count}.jpg"), obj_img)

        # Afficher les occurrences des objets détectés
        y_offset = 30
        for label, count in counts.items():
            cv2.putText(frame, f"{label}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            y_offset += 30

        # Afficher l'image avec les détections
        cv2.imshow('Camera Stream - YOLOv5 Object Detection', frame)

        frame_count += 1
        
        # Sortir de la boucle, si la touche 'q' est pressée
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libérer la capture vidéo et fermer la caméra
    cap.release()
    cv2.destroyAllWindows()

async def main():
    data = await fetch_camera_data()
    if data:
        homes = data.get('body', {}).get('homes', [])
        tasks = []
        for home in homes:
            for camera in home.get('cameras', []):
                print(f"Nom: {camera.get('name')}")
                print(f"VPN URL: {camera.get('vpn_url')}")
                print('---')
                tasks.append(display_camera_streams(camera.get('vpn_url')))
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())

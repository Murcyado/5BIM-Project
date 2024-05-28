import cv2
import numpy as np
from dotenv import load_dotenv
import os
import aiohttp
import asyncio

load_dotenv()

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

async def display_camera_streams(camera_url, net, classes):
    cap = cv2.VideoCapture(camera_url + "/live/files/low/index.m3u8")

    if not cap.isOpened():
        print("Erreur d'ouverture de la caméra réseau")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Erreur dans la capture vidéo")
            break

        detect_and_display(frame, net, classes)

        cv2.imshow('Camera Stream - Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def detect_and_display(frame, net, classes):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.85:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = "{}: {:.2f}%".format(classes[idx], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            save_detected_object(frame, classes[idx], box)

def save_detected_object(frame, label, box):
    (startX, startY, endX, endY) = box
    obj_img = frame[startY:endY, startX:endX]

    dir_path = f'detected_objects/{label}'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    count = len(os.listdir(dir_path))
    img_path = os.path.join(dir_path, f'{label}_{count + 1}.jpg')
    cv2.imwrite(img_path, obj_img)

async def main():
    net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
               "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
               "motorbike", "person", "pottedplant", "sheep", "smartphone", "sofa", "train", "tvmonitor"]

    data = await fetch_camera_data()
    if data:
        homes = data.get('body', {}).get('homes', [])
        tasks = []
        for home in homes:
            for camera in home.get('cameras', []):
                print(f"Nom: {camera.get('name')}")
                print(f"VPN URL: {camera.get('vpn_url')}")
                print('---')
                tasks.append(display_camera_streams(camera.get('vpn_url'), net, CLASSES))
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())

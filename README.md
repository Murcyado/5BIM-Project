Comment lancer le projet : 

Le programme "detect.py" est le programme source servant à lancer le projet.

Il suffit de lancer dans un terminal à partir du dossier ou se trouve ce programme la commande suivante pour lancer le projet : 
py detect.py 

Le fichier "yolov5s.pt" n'est pas à modifier. Il est là seulement pour permettre d'avoir accès au modèle YOLO sur le fichier detect.py.

La classe "Detect_Objet" permet la détéction des objets et l'enregistrement des images des objets détéctés grâce au modèle YoloV5 qui se chargera de la détéction des objets. 
La classe se chargera ensuite de ranger chaque image détécté de l'objet en question (ici laptop et mouse) dans un nouveau dossier nommmé "objets_detectes" et créera d'autres dossiers dans ce dernier pour ranger les images en fonction de l'objet détécté.

La classe "StreamWebcam" va se charger de capturer le flux vidéo générer par la webcam de votre ordinateur et avec l'aide de la classe "Detect_Objet" et du modèle Yolo_V5, de pouvoir détécter les objets en définisant un encadré sur ces mêmes objets avec leur intitulé sur la webcam en temps réel.

import cv2
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import ssl
import os
from dotenv import load_dotenv

load_dotenv()

def send_email(subject, body, image_filename=None):
    user = os.getenv('user')
    password = os.getenv('password')

    recipients = os.getenv('dest')

    sender = user

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = recipients
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # attach img in email
    if image_filename:
        with open(image_filename, 'rb') as image_file:
            image_part = MIMEImage(image_file.read(), name=image_filename)
            msg.attach(image_part)

    context = ssl.create_default_context()

    # send e-mail
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(user, password)
        smtp.sendmail(user, recipients, msg.as_string())


def load_classes(file_path):
    with open(file_path, 'r') as file:
        classes = [line.strip() for line in file.readlines()]
    return classes


def main():
    # Charger la liste des classes (uniquement "personne" dans ce cas)
    classes = load_classes('./class.txt')

    # Charger le modèle YOLO
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

    # Charger les noms des couches de sortie du modèle
    output_layer_names = net.getUnconnectedOutLayersNames()

    # Ouvrir le flux vidéo en temps réel (0 pour la webcam)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur lors de la capture d'un nouveau cadre.")
            break
        
        height, width, channels = frame.shape

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layer_names)

        class_ids = []
        confidences = []
        boxes = []
        for out_detection in outs:
            for detection in out_detection:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5 and classes[class_id] == "person":
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = classes[class_ids[i]]
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                print("personne détectée !")
                cv2.imwrite("personne_detecte.png", frame)
                send_email("Personne détectée !", "Une personne a été détecté. Veuillez vérifier l'activité.", "personne_detecte.png")

        cv2.imshow("Détection d'objets en temps réel", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

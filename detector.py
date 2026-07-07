import cv2
from ultralytics import YOLO
import time
from collections import defaultdict

# Charge le modèle YOLOv8 nano (le plus rapide)
model = YOLO('yolov8n.pt')

# Ouvre la webcam (0 = webcam par défaut)
cap = cv2.VideoCapture(0)

# Compteur d'objets détectés
object_counts = defaultdict(int)

print("Appuyez sur 'q' pour quitter")

while True:
    # Capture frame par frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Détection d'objets
    results = model(frame, verbose=False)
    
    # Dessine les boîtes et compte les objets
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            object_counts[class_name] += 1
            
            # Dessine la boîte
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name} {confidence:.2f}', 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 2)
    
    # Affiche le nombre total de détections
    cv2.putText(frame, f'Total detections: {sum(object_counts.values())}', 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
               1, (255, 0, 0), 2)
    
    # Affiche l'image
    cv2.imshow('Real-time Object Detection', frame)
    
    # Quitte avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libère la webcam et ferme les fenêtres
cap.release()
cv2.destroyAllWindows()

# Affiche le rapport final
print("\n=== RAPPORT DE DÉTECTION ===")
for obj, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{obj}: {count} détections")
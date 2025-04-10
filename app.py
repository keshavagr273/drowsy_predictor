from tensorflow.keras.models import load_model
import os
import gdown
import cv2
import numpy as np

# Download model from Google Drive if not present
model_path = "drowsiness_model.h5"
file_id = "14dpVUBIIZJYRzSQAvuuTeY83dhwawdQL"
gdrive_url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    gdown.download(gdrive_url, model_path, quiet=False)

# Load the model
model = load_model(model_path)

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video
cap = cv2.VideoCapture(0)
image_size = (224, 224)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    label_text = "No Face Detected"
    color = (255, 255, 255)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, image_size)
        face_norm = face_resized / 255.0
        face_input = np.expand_dims(face_norm, axis=0)

        pred = model.predict(face_input)[0]
        label = np.argmax(pred)
        print("Prediction:", pred)

        label_text = "Drowsy" if label == 1 else "Not Drowsy"
        color = (0, 0, 255) if label == 1 else (0, 255, 0)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        break  # only process one face

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

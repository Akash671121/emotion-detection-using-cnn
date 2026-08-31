import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Suppress TensorFlow warnings (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the trained model
model = load_model('emotion.h5')
print("Model Output Shape:", model.output_shape)

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Emotion categories
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Start webcam with reduced resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # ✅ Reduced width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # ✅ Reduced height
cv2.namedWindow("Emotion Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Emotion Detection", 640, 480)

frame_count = 0
last_label = "Detecting..."

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]

        try:
            roi_gray = cv2.resize(roi_gray, (48, 48))
        except:
            continue

        roi = roi_gray.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        if frame_count % 3 == 0:  # ✅ Predict every 3rd frame to reduce flickering
            prediction = model.predict(roi, verbose=0)[0]
            last_label = emotion_labels[np.argmax(prediction)]

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(frame, f"Detected: {last_label}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)

    frame_count += 1
    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

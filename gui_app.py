import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tkinter as tk
from tkinter import Button, Label
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from datetime import datetime
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


model = load_model('emotion.h5')
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


log_file = 'emotion_log.csv'
if not os.path.exists(log_file):
    df = pd.DataFrame(columns=['Timestamp', 'Emotion'])
    df.to_csv(log_file, index=False)


EMOJI_IMAGES = {}
for emotion in EMOTIONS:
    path = f"emojis/{emotion.lower()}.png"
    if os.path.exists(path):
        img = Image.open(path).resize((50, 50))
        EMOJI_IMAGES[emotion] = ImageTk.PhotoImage(img)


emotion_counter = {emotion: 0 for emotion in EMOTIONS}


root = tk.Tk()
root.title("Professional Emotion Detection")
root.geometry("1000x700")
root.configure(bg="#1e1e1e")


detection_running = False
prev_gray = None

cap = cv2.VideoCapture(0)


video_frame = tk.Frame(root, bd=3, relief="sunken", bg="#000000")
video_frame.pack(padx=10, pady=10)
video_label = Label(video_frame, bg="#000000")
video_label.pack()


button_frame = tk.Frame(root, bg="#1e1e1e")
button_frame.pack(pady=15)
btn_font = ("Helvetica", 12, "bold")


def detect_emotions():
    global prev_gray
    if not detection_running:
        return

    ret, frame = cap.read()
    if not ret:
        video_label.after(10, detect_emotions)
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    if prev_gray is not None:
        diff = cv2.absdiff(prev_gray, gray)
        if np.count_nonzero(diff) < 500:
            cv2.putText(frame, "Move face for liveness", (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
            video_label.after(10, detect_emotions)
            return
    prev_gray = gray.copy()

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = model.predict(roi, verbose=0)[0]
        emotion = EMOTIONS[np.argmax(preds)]
        emotion_counter[emotion] += 1

    
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"{emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

       
        if emotion in EMOJI_IMAGES:
            emoji_img = Image.open(f"emojis/{emotion.lower()}.png").resize((50,50))
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_pil.paste(emoji_img, (x, y-50), mask=emoji_img)
            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

      
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df = pd.DataFrame([[timestamp, emotion]], columns=['Timestamp', 'Emotion'])
        df.to_csv(log_file, mode='a', header=False, index=False)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    video_label.after(10, detect_emotions)

def toggle_detection():
    global detection_running
    if not detection_running:
        detection_running = True
        start_btn.config(text="Stop Detection", bg='#dc3545')
        detect_emotions()
    else:
        detection_running = False
        start_btn.config(text="Start Detection", bg='#28a745')

def show_confusion_matrix():
    if not os.path.exists(log_file):
        print("No CSV file found!")
        return
    df = pd.read_csv(log_file)
    if df.empty:
        print("CSV is empty!")
        return

    y_true = df['Emotion']
    y_pred = df['Emotion']  

    cm = confusion_matrix(y_true, y_pred, labels=EMOTIONS)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=EMOTIONS)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.pdf")
    plt.show()
    print("Confusion matrix saved as confusion_matrix.pdf")


def on_enter(btn, color): btn['bg'] = color
def on_leave(btn, color): btn['bg'] = color


start_btn = tk.Button(button_frame, text="Start Detection", command=toggle_detection,
                      bg='#28a745', fg='white', width=18, height=2, font=btn_font, relief="raised", bd=3)
start_btn.pack(side="left", padx=15)
start_btn.bind("<Enter>", lambda e: on_enter(start_btn, "#218838"))
start_btn.bind("<Leave>", lambda e: on_leave(start_btn, "#28a745"))

analytics_btn = tk.Button(button_frame, text="Show Confusion Matrix", command=show_confusion_matrix,
                          bg='#007bff', fg='white', width=22, height=2, font=btn_font, relief="raised", bd=3)
analytics_btn.pack(side="left", padx=15)
analytics_btn.bind("<Enter>", lambda e: on_enter(analytics_btn, "#0069d9"))
analytics_btn.bind("<Leave>", lambda e: on_leave(analytics_btn, "#007bff"))

exit_btn = tk.Button(button_frame, text="Exit", command=root.destroy,
                     bg='#dc3545', fg='white', width=18, height=2, font=btn_font, relief="raised", bd=3)
exit_btn.pack(side="left", padx=15)
exit_btn.bind("<Enter>", lambda e: on_enter(exit_btn, "#c82333"))
exit_btn.bind("<Leave>", lambda e: on_leave(exit_btn, "#dc3545"))


root.mainloop()
cap.release()
cv2.destroyAllWindows()

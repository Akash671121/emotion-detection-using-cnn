# Emotion Detection Using CNN

## Overview

This project is a real-time facial emotion detection system developed using Python, Convolutional Neural Network (CNN), OpenCV, and Tkinter.

The system uses a webcam to detect a person's face and predicts one of seven emotions:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

## Features

- Real-time emotion detection using a webcam
- Face detection using Haar Cascade
- CNN-based emotion classification
- Graphical User Interface (GUI)
- Emotion prediction logging
- Confusion matrix for model evaluation

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Tkinter
- Matplotlib
- Scikit-learn

## How It Works

1. The webcam captures the video.
2. OpenCV detects the face using Haar Cascade.
3. The detected face is converted to grayscale.
4. The image is resized to 48 × 48 pixels and normalized.
5. The trained CNN model predicts the emotion.
6. The predicted emotion is displayed in real time.

## Project Structure

```text
emotion-detection-using-cnn/
│
├── gui_app.py
├── train_model.py
├── video_test.py
├── emotion.h5
├── haarcascade_frontalface_default.xml
├── confusion_matrix.pdf
├── emotion_log.csv
├── README.md
└── .gitignore
```
## Model

The project uses a Convolutional Neural Network (CNN) trained to classify facial expressions into seven emotion categories.

The trained model is stored in:

`emotion.h5`

## Results

The model performance was evaluated using a confusion matrix.

See `confusion_matrix.pdf` for the evaluation result.

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Akash671121/emotion-detection-using-cnn.git
```

### 2. Install the required libraries
```bash
pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn pillow
```

### 3. Run the application
```text
python gui_app.py
```

## Future Improvements
- Improve model accuracy
- Add more emotion categories
- Improve real-time prediction performance
- Deploy the application as a web application
- Add better visualization and reporting

## Author

***Akash K***

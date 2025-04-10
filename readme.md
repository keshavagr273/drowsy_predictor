# Drowsiness Predictor

This project is a **Drowsiness Predictor** that uses a face photo to detect whether a person is drowsy or not. It leverages a pre-trained deep learning model and OpenCV for real-time face detection and prediction.

## Features
- Real-time face detection using OpenCV's Haar Cascade Classifier.
- Drowsiness prediction using a TensorFlow/Keras model.
- Automatic model download from Google Drive if not already present.

## How It Works
1. The application captures video from the webcam.
2. It detects faces in the video feed using OpenCV's Haar Cascade Classifier.
3. For each detected face, the model predicts whether the person is drowsy or not.
4. The result is displayed on the video feed with a bounding box and label.

## Technologies Used
- **Python**: Programming language used for the project.
- **TensorFlow/Keras**: For loading and using the pre-trained deep learning model.
- **OpenCV**: For real-time face detection and video processing.
- **NumPy**: For numerical operations and data preprocessing.
- **gdown**: For downloading the model file from Google Drive.

## Model
The pre-trained model used for drowsiness prediction is hosted on Google Drive. It will be automatically downloaded when you run the application. You can also manually download it from the following link:

[Download Model](https://drive.google.com/file/d/14dpVUBIIZJYRzSQAvuuTeY83dhwawdQL/view?usp=sharing)

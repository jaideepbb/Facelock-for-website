from flask import Flask, request, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load a pre-registered face image and encode it
registered_image_path = r'C:\Users\jalaj\OneDrive\Desktop\vs code ap\face-lock\easy.jpg'
if not os.path.exists(registered_image_path):
    raise FileNotFoundError(f"Registered image file '{registered_image_path}' not found.")

registered_image = cv2.imread(registered_image_path)
if registered_image is None:
    raise ValueError(f"Failed to load image file '{registered_image_path}'.")

registered_gray = cv2.cvtColor(registered_image, cv2.COLOR_BGR2GRAY)

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "No file uploaded"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "message": "No file selected"})

    # Save the uploaded file temporarily
    file_path = "uploaded_face.jpg"
    file.save(file_path)

    # Load the uploaded image
    uploaded_image = cv2.imread(file_path)
    if uploaded_image is None:
        os.remove(file_path)
        return jsonify({"success": False, "message": "Failed to load uploaded image"})

    uploaded_gray = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the uploaded image
    faces = face_cascade.detectMultiScale(uploaded_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        os.remove(file_path)
        return jsonify({"success": False, "message": "No face detected in the image"})

    # Compare the detected face with the registered face
    (x, y, w, h) = faces[0]
    uploaded_face = uploaded_gray[y:y+h, x:x+w]
    registered_face_resized = cv2.resize(registered_gray, (w, h))

    # Simple comparison using Mean Squared Error (MSE)
    mse = np.mean((registered_face_resized - uploaded_face) ** 2)
    os.remove(file_path)  # Delete the uploaded file after processing

    if mse < 5000:  # Adjust threshold as needed
        return jsonify({"success": True, "message": "Face recognized"})
    else:
        return jsonify({"success": False, "message": "Face not recognized"})

if __name__ == '__main__':
    app.run(debug=True)

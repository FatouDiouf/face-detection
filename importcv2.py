import streamlit as st
import numpy as np
import cv2

def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def detect_faces_video(video_file):
    video_file = video_file.name  # Récupérer le nom du fichier
    video = cv2.VideoCapture(video_file)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        faces = detect_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        st.image(frame, channels="BGR", caption="Visages détectés")
    video.release()

def main():
    st.title("Détection de visages")
    option = st.sidebar.selectbox("Choisissez le mode", ("Photo", "Vidéo"))

    if option == "Photo":
        uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
            faces = detect_faces(image)
            st.image(image, channels="BGR", caption="Image originale")
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            st.image(image, channels="BGR", caption="Visages détectés")

    elif option == "Vidéo":
        video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", "avi", "asf", "m4v"])
        if video_file_buffer is not None:
            st.video(video_file_buffer)
            detect_faces_video(video_file_buffer)

if __name__ == '__main__':
    main()


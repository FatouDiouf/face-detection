import streamlit as st  # Importe la bibliothèque Streamlit pour créer des applications Web
import numpy as np  # Importe la bibliothèque numpy pour les opérations mathématiques
import cv2  # Importe la bibliothèque OpenCV pour la vision par ordinateur

# Cette fonction détecte les visages dans une image donnée
def detect_faces(image):
    # Charge le classificateur pré-entraîné pour la détection de visage
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Convertit l'image en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Détecte les visages dans l'image en niveaux de gris
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces  # Retourne les coordonnées des visages détectés

# Cette fonction détecte les visages dans une vidéo donnée
def detect_faces_video(video_file):
    # Récupère le nom du fichier vidéo
    video_file = video_file.name  
    # Ouvre le fichier vidéo
    video = cv2.VideoCapture(video_file)
    # Lit la vidéo frame par frame
    while True:
        ret, frame = video.read()
        # Si la lecture de la vidéo échoue, interrompt la boucle
        if not ret:
            break
        # Détecte les visages dans chaque frame
        faces = detect_faces(frame)
        # Dessine un rectangle vert autour de chaque visage détecté
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Affiche l'image avec les visages détectés
        st.image(frame, channels="BGR", caption="Visages détectés")
    # Libère le fichier vidéo
    video.release()

# Le point d'entrée principal du programme
def main():
    # Affiche le titre de l'application
    st.title("Détection de visages")
    # Crée un menu déroulant pour choisir entre la détection de visages sur une photo ou une vidéo
    option = st.sidebar.selectbox("Choisissez le mode", ("Photo", "Vidéo"))

    # Si l'option "Photo" est sélectionnée
    if option == "Photo":
        # Permet à l'utilisateur de télécharger une image
        uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])
        # Si une image a été téléchargée
        if uploaded_file is not None:
            # Convertit l'image téléchargée en un tableau numpy
            image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
            # Détecte les visages dans l'image
            faces = detect_faces(image)
            # Affiche l'image originale
            st.image(image, channels="BGR", caption="Image originale")
            # Dessine un rectangle vert autour de chaque visage détecté
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Affiche l'image avec les visages détectés
            st.image(image, channels="BGR", caption="Visages détectés")

    # Si l'option "Vidéo" est sélectionnée
    elif option == "Vidéo":
        # Permet à l'utilisateur de télécharger une vidéo
        video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", "avi", "asf", "m4v"])
        # Si une vidéo a été téléchargée
        if video_file_buffer is not None:
            # Affiche la vidéo originale
            st.video(video_file_buffer)
            # Détecte les visages dans la vidéo
            detect_faces_video(video_file_buffer)

# Si le script est exécuté directement (et non importé comme un module), exécute la fonction principale
if __name__ == '__main__':
    main()

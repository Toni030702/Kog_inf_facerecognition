import cv2
import mediapipe as mp

# Inicijalizacija MediaPipe Face Detection modela
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.3)

# Inicijalizacija MediaPipe Face Mesh modela
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Funkcija za obradu slike ili videa
def process_image_or_video(input_path, output_path=None, is_video=False):
    if is_video:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print("Greska")
            return
    else:
        img = cv2.imread(input_path)
        if img is None:
            print("Greska")
            return

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))

    # Obrada frame-ova
    while True:
        if is_video:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame = img.copy()

        # Pretvorba slike u RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detekcija lica
        results_face = face_detection.process(rgb_frame)

        # Crtanje kvadrata oko detektiranih lica
        if results_face.detections:
            for detection in results_face.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(frame, bbox, (0, 255, 0), 2)

                # Ispisivanje pouzdanosti
                if hasattr(detection, 'score'):
                    confidence = int(detection.score[0] * 100)
                    cv2.putText(frame, f"Confidence: {confidence}%",
                                (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Detekcija Face Mesh-a
        results_mesh = face_mesh.process(rgb_frame)

        # Crtanje Face Mesh-a
        if results_mesh.multi_face_landmarks:
            for face_landmarks in results_mesh.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    x, y = int(landmark.x * iw), int(landmark.y * ih)
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), 1)

        # Prikazivanje rezultata
        cv2.imshow("Face Detection", frame)
        if output_path:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Esc
            break

    if is_video:
        cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

# Primjeri korištenja

# Za obradu slike i spremanje rezultata
process_image_or_video('Images/FR_slika.jpeg')

# Za obradu videa i prikazivanje rezultata
process_image_or_video("Videos/FR_video.mp4", is_video=True)

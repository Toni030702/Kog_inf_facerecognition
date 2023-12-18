import cv2
import mediapipe as mp

# Inicijalizacija MediaPipe Face Detection modela
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.3)

# Inicijalizacija MediaPipe Face Mesh modela
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Funkcija za obradu slike ili videa s kamerom
def process_image_or_video(input_path=None, output_path=None, is_video=False):
    if is_video or input_path is None:  # Koristi kameru za video ili ako nije specificiran ulazni put
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Greska")
            return
    else:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print("Greska")
            return

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))

    # Obrada frame-ova
    while True:
        ret, frame = cap.read()
        if not ret:
            break

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

    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()


process_image_or_video()
process_image_or_video(f'Videos/FR_video.mp4')

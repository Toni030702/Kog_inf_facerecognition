from deepface import DeepFace
import cv2

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

    while True:
        if is_video:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame = img.copy()

        # Koristi DeepFace za analizu slike
        result = DeepFace.analyze(frame, actions=['age', 'gender'])

        # Dodaj informacije na sliku
        info_text = f"Age: {result['age']}, Gender: {result['gender']}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Face Analysis", frame)
        if output_path:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    if is_video:
        cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

# Primjer korištenja
process_image_or_video(f'Images/FR_slika.jpeg')

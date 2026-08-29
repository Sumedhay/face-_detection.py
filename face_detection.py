import cv2

# Loading the Haar Cascade files
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# Start the webcam
camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()

    if not success:
        print("Unable to access the camera")
        break

    # Convert the frame into grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find faces
    faces = face_cascade.detectMultiScale(
        gray,
        1.1,
        5
    )

    for x, y, w, h in faces:

        # Draw a rectangle around the face
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        # Look for eyes inside the detected face
        face_gray = gray[y:y + h, x:x + w]
        face_frame = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(
            face_gray,
            1.1,
            5
        )

        for ex, ey, ew, eh in eyes:

            cv2.rectangle(
                face_frame,
                (ex, ey),
                (ex + ew, ey + eh),
                (0, 255, 0),
                2
            )

    # Show the webcam output
    cv2.imshow("Face and Eye Detection", frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

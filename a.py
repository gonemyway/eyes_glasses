import cv2
import numpy as np

face_detection_model = cv2.CascadeClassifier("models/haarcascade_frontalface_alt.xml")
eyes_detection_model = cv2.CascadeClassifier("models/haarcascade_eye.xml")

glass_image = cv2.imread("glass_image/glasses_01.png", -1)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if ret:
        final_image = frame

        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detection_model.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5, minSize=(200, 200))
        if len(faces):
            for (face_x, face_y, face_w, face_h) in faces:
                face = gray_image[face_y: face_y + face_h, face_x: face_x + face_w] # Phát hiện từng khuôn mặt
                eyes = eyes_detection_model.detectMultiScale(face, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                eyes_center = []
                for (eye_x, eye_y, eye_w, eye_h) in eyes:
                    eyes_center.append((face_x + int(eye_x + eye_w/2), face_y + int(eye_y + eye_h/2)))

                if len(eyes_center) >= 2:
                    glass_width = 2.5 * abs(eyes_center[1][0] - eyes_center[0][0])
                    scale_factor = glass_width / glass_image.shape[1]
                    glass_resize = cv2.resize(glass_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

                    if eyes_center[0][0] < eyes_center[1][0]:
                        left_eye_x = eyes_center[0][0]
                    else:
                        left_eye_x = eyes_center[1][0]

                    # Tọa độ của kính
                    glass_x = int(left_eye_x - 0.25 * glass_resize.shape[1])
                    glass_y = int(face_y + 0.85 * glass_resize.shape[0])

                    # Kiểm tra xem kính có nằm trong khung hình không
                    if 0 <= glass_x < frame.shape[1] and 0 <= glass_y < frame.shape[0] and \
                       glass_x + glass_resize.shape[1] < frame.shape[1] and \
                       glass_y + glass_resize.shape[0] < frame.shape[0]:

                        for i in range(glass_resize.shape[0]):
                            for j in range(glass_resize.shape[1]):
                                if glass_resize[i, j][3] != 0:  # kênh alpha != 0
                                    frame[glass_y + i, glass_x + j] = glass_resize[i, j]

        cv2.imshow("Glasses", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

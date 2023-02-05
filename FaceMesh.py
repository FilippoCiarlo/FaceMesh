import cv2
import mediapipe as mp
import time

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("Videos/2.mp4")
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMash = mp.solutions.face_mesh
faceMesh = mpFaceMash.FaceMesh(static_image_mode=False,
                               max_num_faces=2,
                               refine_landmarks=False,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

while True:
    # read image
    success, img = cap.read()

    # convert the image from BGR to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # process the given image
    results = faceMesh.process(imgRGB)

    # diplay landmarks
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMash.FACEMESH_CONTOURS,
                                  drawSpec,drawSpec)
            for id,lm in enumerate(faceLms.landmark):
                ih, iw, ic = img.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                print(id,x,y)

    # calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # visualize output
    cv2.putText(img, f"FPS:{str(int(fps))}", (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (80, 255, 222), 2)
    cv2.imshow('FaceMesh', img)
    cv2.waitKey(1)
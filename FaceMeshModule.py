import cv2
import mediapipe as mp
import time




class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=1, refineLms=False, minDetectCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.refineLms = refineLms
        self.minDetectCon = minDetectCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMash = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMash.FaceMesh(self.staticMode,
                                                 self.maxFaces,
                                                 self.refineLms,
                                                 self.minDetectCon,
                                                 self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw=True):
        # convert the image from BGR to RGB
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # process the given image
        self.results = self.faceMesh.process(self.imgRGB)

        # diplay landmarks
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMash.FACEMESH_CONTOURS,self.drawSpec,self.drawSpec)

                face = []
                for id,lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    #cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.5, (80, 255, 222), 1)
                    face.append([x,y])
                faces.append(face)

        return img, faces

def main():
    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture("Videos/2.mp4")
    pTime = 0
    detector = FaceMeshDetector(maxFaces=2)

    while True:
        # read image
        success, img = cap.read()

        # detect face
        img, faces = detector.findFaceMesh(img,draw=True)
        if len(faces) != 0:
            print(faces[0])

        # calculate FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # visualize output
        cv2.putText(img, f"FPS:{str(int(fps))}", (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (80, 255, 222), 2)
        cv2.imshow('FaceMesh', img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
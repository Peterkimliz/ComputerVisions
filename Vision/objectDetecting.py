import cv2
video = cv2.VideoCapture("peterAi.mkv")
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
while True:
    check, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = faceCascade.detectMultiScale(gray, scaleFactor=1.25, minNeighbors=5)

    for x, y, w, h in face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(frame, "human", (x, y-2), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow("Video", frame)
    key = cv2.waitKey(1)
    if key == ord("r"):
        break
video.release()
cv2.destroyAllWindows()





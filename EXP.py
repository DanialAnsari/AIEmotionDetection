import cv2 



face_classifier = cv2.CascadeClassifier('data\haarcascade_frontalface_alt2.xml')
smile_classifier = cv2.CascadeClassifier('data\haarcascade_smile.xml')
eye_classifier = cv2.CascadeClassifier('data\haarcascade_lefteye_2splits.xml')
times=[]
smile_ratios=[]
cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    faces = face_classifier.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
        roi_gray = gray[y:y + h, x:x + w]
        roi_img = img[y:y + h, x:x + w]
        smile = smile_classifier.detectMultiScale(roi_gray, scaleFactor=1.2,
                                                  minNeighbors=22,
                                                  minSize=(25, 25))
        eye = eye_classifier.detectMultiScale(roi_gray, scaleFactor=1.2,
                                                  minNeighbors=22,
                                                  minSize=(25, 25))
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(roi_img, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 1)
            sm_ratio = round(sw / sx, 3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            break
            
        for (ex, ey, ew, eh) in eye:
            cv2.rectangle(roi_img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)
            em_ratio = round(ew / ey, 3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            break
        ratio=10000*sm_ratio/em_ratio
        emo=""
        if(ratio<15000):
            emo="Poker"
        elif(ratio>=15000 and ratio<17500):
            emo="Sad"
        elif(ratio>=17500 and ratio<20000):
            emo="Angry"
        elif(ratio>=20000):
            emo="Smile"
        cv2.putText(img, emo+' : ' + str(ratio), (10, 50), font, 1, (200, 255, 155), 2, cv2.LINE_AA)
    cv2.imshow('Smile Detector', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

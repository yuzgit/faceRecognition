#--------------------------------------------
# カスケード分類機を用いた顔認識
#--------------------------------------------

import cv2

img = cv2.imread('images/face_img02.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# カスケードファイルの読み込み
# カスケードファイル: 顔とそれ以外を分類できる情報が格納されているXMLファイル
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#カスケードファイルを使って顔認証
faces = face_cascade.detectMultiScale(gray)
for (x,y,w,h) in faces:
    #顔部分を四角で囲う
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
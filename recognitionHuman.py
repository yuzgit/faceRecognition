#----------------------------------------------------------------------------------------------
# 人間検出(HoG + SVM識別器)
# HoG特徴量: 領域内の勾配方向ごとの勾配強度を計算しヒストグラムで表したもの
# SVM(サポートベクターマシン): データの境界線を見つけデータの分類を行う手法
# 参考URL: https://algorithm.joho.info/programming/python/opencv-hog-cascade-human-detection-py/
#----------------------------------------------------------------------------------------------

import cv2 

img = cv2.imread('images/human_img01.jpg')

#グレースケールへ変換
#RGBのままだと色情報に精度を左右されやすいのでグレースケールにして誤差を減らす
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#HoG特徴量 + SVMで人の判別機の作成
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
hogParams = {'winStride':(8,8), 'padding':(32,32), 'scale':1.05}

#判別機を用いて人の検出
human,r = hog.detectMultiScale(gray,**hogParams)

#人の領域を囲う
for (x,y,w,h) in human:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

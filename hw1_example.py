# -*- coding: utf-8 -*-

import sys
from hw1_ui import Ui_MainWindow
import numpy as np
import cv2
import math
from PyQt5.QtWidgets import QMainWindow, QApplication, QLineEdit

dog_img = cv2.imread('image\dog.bmp')
dog_flip = cv2.flip(dog_img, 1)
M8_img = cv2.imread('image\M8.jpg')
qr_img = cv2.imread('image\QR.png')
sliderValue = 0


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.onBindingUI()

    # Write your code below
    # UI components are defined in hw1_ui.py, please take a look.
    # You can also open hw1.ui by qt-designer to check ui components.

    def onBindingUI(self):
        self.btn1_1.clicked.connect(self.on_btn1_1_click)
        self.btn1_2.clicked.connect(self.on_btn1_2_click)
        self.btn1_3.clicked.connect(self.on_btn1_3_click)
        self.btn1_4.clicked.connect(self.on_btn1_4_click)
        self.btn2_1.clicked.connect(self.on_btn2_1_click)
        self.btn3_1.clicked.connect(self.on_btn3_1_click)
        self.btn4_1.clicked.connect(self.on_btn4_1_click)
        self.btn4_2.clicked.connect(self.on_btn4_2_click)
        self.btn5_1.clicked.connect(self.on_btn5_1_click)
        self.btn5_2.clicked.connect(self.on_btn5_2_click)

    # button for problem 1.1
    def on_btn1_1_click(self):
        #print("Hello world!")  # add your code here
        #dog_img = cv2.imread('image\dog.bmp')
        cv2.namedWindow("1.1 Load img")
        cv2.imshow("1.1 Load img",dog_img)
        #print(dog_img.shape)
        print("Height = ",dog_img.shape[0])
        print("Weight = ",dog_img.shape[1])
        cv2.waitKey (0)
        

    def on_btn1_2_click(self):
        #print("This is btn1_2")
        color_img = cv2.imread('image\color.png')
        cv2.namedWindow("1.2 color.png")
        cv2.imshow("1.2 color.png",color_img)
        img = color_img
        (B,G,R) = cv2.split(img)#提取R、G、B分量
        img = cv2.merge([G,R,B])#合并R、G、B分量
        cv2.namedWindow("1.2 color change")
        cv2.imshow("1.2 color change",img)
        cv2.waitKey(0)


    def on_btn1_3_click(self):
        #pass
        #dog_img = cv2.imread('image\dog.bmp')
        cv2.namedWindow("1.3 dog.bmp")
        cv2.imshow("1.3 dog.bmp",dog_img)
        dog_flip = cv2.flip(dog_img, 1)
        cv2.imshow("Dog flip", dog_flip)


    def on_btn1_4_click(self):
        def on_trackbar(sv):
            ratio = sv/100.0
            mix_dog = cv2.addWeighted(dog_img, ratio, dog_flip,(1.0 - ratio), 0.0)
            cv2.imshow("1.4 mix_dog", mix_dog)
        #pass
        sliderMaxValue = 100
        cv2.namedWindow("1.4 mix_dog")
        cv2.createTrackbar("Mix ratio", "1.4 mix_dog", sliderValue, sliderMaxValue, on_trackbar)
        on_trackbar(sliderValue)
        cv2.waitKey(0)


    def on_btn2_1_click(self):
        #pass
        M8_gray = cv2.cvtColor(M8_img, cv2.COLOR_BGR2GRAY)#RGB to grayscale
        #cv2.imshow("M8_gray",M8_gray)
        M8_blurred = cv2.GaussianBlur(M8_gray, (3, 3), 0)#Gaussian smooth
        #cv2.imshow("M8_blurred",M8_blurred)

        #sobel
        def xGradient(image, x, y):
            return (image[x-1][y-1] + 2*image[x-1][y] +
                    image[x-1][y+1] - image[x+1][y-1] -
                    2*image[x+1][y] - image[x+1][y+1])
        def yGradient(image, x, y):
            return (image[x-1][y-1] + 2*image[x][y-1] +
                    image[x+1][y-1] - image[x-1][y+1] -
                    2*image[x][y+1] - image[x+1][y+1])
 
        src = M8_blurred
        dstx = M8_blurred.copy()
        dsty = M8_blurred.copy()
        G = M8_blurred.copy()
        angle = M8_blurred.copy()
        for y in range(0,src.shape[0]):
            for x in range(0,src.shape[1]):
                dstx[y][x] = 0
                dsty[y][x] = 0
                G[y][x] = 0
                angle[y][x] = 0
        for x in range(1,src.shape[0]-1):
             for y in range(1,src.shape[1]-1):
                gx = xGradient(src, x, y)
                gy = yGradient(src, x, y)
                if(abs(gx)>100):
                    sum_x = 255
                else:
                    sum_x = 0
                if(abs(gy)>100):
                    sum_y = 255
                else:
                    sum_y = 0
                dstx[x][y] = sum_x
                dsty[x][y] = sum_y
                if(gx == 0):
                    div = 0
                else:
                    div = gy/gx
                angle[x][y] = math.degrees(math.atan((div)))
                G[x][y] = np.sqrt(np.square(gx) + np.square(gy))
        cv2.namedWindow("2.3 Soble in the x direction")
        cv2.imshow("2.3 Soble in the x direction",dstx)
        cv2.namedWindow("2.3 Soble in the y direction")
        cv2.imshow("2.3 Soble in the y direction",dsty)
        def on_trackbar(sv):
            Gm = G.copy()
            for x in range(1,src.shape[0]-1):
                for y in range(1,src.shape[1]-1):
                    if(G[x][y] > sv):
                        Gm[x][y] = 255
                    else:
                        Gm[x][y] = 0
            cv2.imshow("Magnitude", Gm)
        sliderValue = 40
        sliderMaxValue = 255
        cv2.namedWindow("Magnitude")
        cv2.createTrackbar("threshold", "Magnitude", sliderValue, sliderMaxValue, on_trackbar)
        on_trackbar(sliderValue)

        #angle
        def on_trackbar(sv):
            Ga = G.copy()
            for x in range(1,src.shape[0]-1):
                for y in range(1,src.shape[1]-1):
                    if(angle[x][y] > sv+10 or angle[x][y] < sv-10):
                        Ga[x][y] = 0
                    else:
                        Ga[x][y] = 255
            cv2.imshow("angle", Ga)
        sliderValue = 10
        sliderMaxValue = 360
        cv2.namedWindow("angle")
        cv2.createTrackbar("angle", "angle", sliderValue, sliderMaxValue, on_trackbar)
        on_trackbar(sliderValue)
        cv2.waitKey(0)


    def on_btn3_1_click(self):
        #pass
        pyr_img = cv2.imread('image\pyramids_Gray.jpg',0)
        #cv2.imshow("pyramid level0",pyr_img)
        pyr_level0_s = cv2.GaussianBlur(pyr_img, (3, 3), 0)#Gaussian smooth
        pyr_level1 = cv2.pyrDown(pyr_level0_s)
        cv2.imshow("pyramid level1",pyr_level1)
        #Laplacian pyramid level 0 image
        pyr_level1_up = cv2.pyrUp(pyr_level1)
        pyr_level1_up = cv2.GaussianBlur(pyr_level1_up, (3, 3), 0)#Gaussian smooth
        la_level0 = cv2.subtract(pyr_img, pyr_level1_up)
        cv2.imshow("Laplacian pyramid of level 0",la_level0)
        pyr_level1_s = cv2.GaussianBlur(pyr_level1, (3, 3), 0)#Gaussian smooth
        pyr_level2 = cv2.pyrDown(pyr_level1_s)
        #Laplacian pyramid level 1 image
        pyr_level2_up = cv2.pyrUp(pyr_level2)
        pyr_level2_up = cv2.GaussianBlur(pyr_level2_up, (3, 3), 0)#Gaussian smooth
        la_level1 = cv2.subtract(pyr_level1, pyr_level2_up)
        #Inverse pyramid level 1 image
        inv_level1 = cv2.add(pyr_level2_up, la_level1)
        cv2.imshow("Inverse level 1",inv_level1)
        #Inverse pyramid level 0 image
        inv_level0 = pyr_level1_up + la_level0
        cv2.imshow("Inverse level 0",inv_level0)


    def on_btn4_1_click(self):
        #pass
        cv2.imshow("Original image",qr_img)
        qr_gray = cv2.cvtColor(qr_img, cv2.COLOR_BGR2GRAY)
        ret,g_thresh = cv2.threshold(qr_gray,75,255,cv2.THRESH_BINARY)
        cv2.imshow("Global threshold",g_thresh)


    def on_btn4_2_click(self):
        #pass
        cv2.imshow("Original image",qr_img)
        qr_gray = cv2.cvtColor(qr_img, cv2.COLOR_BGR2GRAY)
        l_thresh = cv2.adaptiveThreshold(qr_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,19,-1)
        cv2.imshow("Local threshold",l_thresh)


    def on_btn5_1_click(self):
        # edtAngle, edtScale. edtTx, edtTy to access to the ui object
        angle = float(self.edtAngle.text())
        scale = float(self.edtScale.text())
        tx = float(self.edtTx.text())
        ty = float(self.edtTy.text())
        tran_img = cv2.imread('image\OriginalTransform.png')
        cv2.imshow("Original image",tran_img)
        #rotate, scale
        h, w = tran_img.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(tran_img, M, (w, h))
        #translate
        M = np.float32([[1, 0, tx+150], [0, 1, ty+50]])
        shifted = cv2.warpAffine(rotated, M, (w, h))
        cv2.imshow("Transform image",shifted)

    
    def on_btn5_2_click(self):
        per_img = cv2.imread('image\OriginalPerspective.png')
        cv2.namedWindow('Original image')
        cv2.imshow("Original image",per_img)
        px = []
        py = []
        def onMouse(Event, x, y, flags, param):
            if(Event==cv2.EVENT_LBUTTONDOWN):
                px.append(x)
                py.append(y)
                print(px,py)
                if(len(px) == 4):
                    pos = []
                    for i in range(0,4):
                        pos.append([px[i],py[i]])
                    print(pos)
                    (tl, tr, br, bl) = (pos[0],pos[1],pos[2],pos[3])
                    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                    maxWidth = max(int(widthA), int(widthB))
                    maxHeight = max(int(heightA), int(heightB))
                    dst = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]], dtype = "float32")
                    pos = np.array(pos,dtype = "float32")
                    M = cv2.getPerspectiveTransform(pos, dst)
                    warp = cv2.warpPerspective(per_img, M, (maxWidth, maxHeight))
                    cv2.imshow("Perspective Image",warp)

        cv2.setMouseCallback('Original image',onMouse)


    ### ### ###


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 09:07:18 2018

@author: chonlatid.d
"""

import sys
import cv2
from PyQt5.QtCore import QTimer
#from PyQt5 import uic, QtWidgets
from PyQt5.QtWidgets import QApplication , QMainWindow
from PyQt5.uic import loadUi
from PyQt5.QtGui import QPixmap, QImage       


import re
import numpy as np
from keras.models import load_model
from PIL import Image, ImageDraw , ImageOps        
import pytesseract
pytesseract.pytesseract.tesseract_cmd= r'Tesseract-OCR\tesseract.exe'
import time
#qtCreatorFile = "webcamshow.ui" # Enter file here.
#Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

class WEBCAMUI(QMainWindow):
    def __init__(self):
        super(WEBCAMUI,self).__init__()
        loadUi('mainform.ui',self)
        
        self.image=None
        self.toggle_flag = False
        self.startButton.clicked.connect(self.start_check)
        self.stopButton.clicked.connect(self.stop_webcam)
        self.setAcceptDrops(True)
        self.crop_size = (125*4,200*4)
        self.w = 640       
        self.h = 480
        self.crop_size = (0.3*self.h,0.35*self.w)
       
        self.model = load_model(r'kengshawguay.h5')
        self.score_model = load_model('ModelDisV2.h5')
        self.model_num = load_model(r'numcrop_model.h5')
        self.model_ocr = load_model(r'classifier_modelv3.h5')
        self.alphabet = u'0123456789'
        
        
    
    def plot_corner(self,img,predict_corner,color = (255,0,0)):
        ret_img = (img)*255
        ret_img = Image.fromarray(ret_img.astype('uint8'))
        
        crop_numim = np.asarray(ret_img)
        crop_numim = crop_numim[int(predict_corner[1]):int(predict_corner[3]) , int(predict_corner[0]):int(predict_corner[2])]
        
        crop_numim = cv2.cvtColor(crop_numim, cv2.COLOR_BGR2RGB) 
        cv2.imwrite('crop_numim.png',crop_numim)
        
        draw = ImageDraw.Draw(ret_img)
        draw.rectangle([predict_corner[0],predict_corner[1],predict_corner[2],predict_corner[3] ],  outline=(255,0,0))
        
        return ret_img
    
    def order_points(self,pts):

    	rect = np.zeros((4, 2), dtype="float32")
    
    	s = pts.sum(axis=1)
    	rect[0] = pts[np.argmin(s)]
    	rect[2] = pts[np.argmax(s)]
    
    
    	diff = np.diff(pts, axis=1)
    	rect[1] = pts[np.argmin(diff)]
    	rect[3] = pts[np.argmax(diff)]
   
    	return rect
    
    def warpIDcard(self,path,mode = 'pic'):

        image0 = Image.open(path)
        image0 = image0.convert('RGB')
        
        image1 = np.asarray(image0)
        
        if(mode == 'cam'):
            h,w,d= image1.shape
            crop_size = (0.3*h,0.25*w)
            image1 = image1[int(h/2 - crop_size[0]/2):int(h/2 + crop_size[0]/2) , int(w/2 - crop_size[1]/2):int(w/2 + crop_size[1]/2)]
            image0 = Image.fromarray(image1.astype('uint8'))
            
            
        imsize = 256 
        image1 = cv2.resize(image1,(imsize,imsize))
        image1 = image1 / 127.5 - 1
        image1 = np.expand_dims(image1,axis=0)
        foreground = self.model.predict(image1)
        
        corner = np.reshape(foreground,(4,2))
        
        ratio_h = image0.height / imsize
        ratio_w = image0.width / imsize
        image2 = np.asarray(image0)
        orig = image2.copy()
        
        corner[:,0] *= ratio_w  
        corner[:,1] *= ratio_h
        
        image0 = np.asarray(image0)
        drawing = image0.astype(np.uint8)
        drawing = np.squeeze(drawing)
        
        c = corner.astype(np.int)
        c = np.reshape(c,(1,4,2))
        for i in range(0,1):
            color_contours = (0, 255, 0) # green - color for contours
            cv2.drawContours(drawing,  c, -1, color_contours, 1, 8)
            
        screenCnt = c
        warped = self.transformFourPoints(orig, screenCnt.reshape(4, 2))
        warped = Image.fromarray(warped.astype('uint8'),'RGB')
        warped.save('test_crop.png')

    def IDscore(self):
        image = Image.open('test_crop.png')

        image = image.resize((128,128))
        image = np.asarray(image)
        image = image / 127.5 - 1.
        image = np.expand_dims(image,axis=0)
        resultr=self.score_model.predict(image) #use model
        
        return resultr
    
    def transformFourPoints(self,image, pts):
        
    	rect = self.order_points(pts)
    	(tl, tr, br, bl) = rect
        
    	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    	maxWidth = max(int(widthA), int(widthB))
    
    
    	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    	maxHeight = max(int(heightA), int(heightB))
        
    	dst = np.array([[0, 0],	[maxWidth - 1, 0],	[maxWidth - 1, maxHeight - 1],	[0, maxHeight - 1]], dtype="float32")
        
    	M = cv2.getPerspectiveTransform(rect, dst)
    	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    	return warped
    
    def dragEnterEvent(self, e):
      
#        if e.mimeData().hasFormat('text/plain'):
        e.accept()
            
#        else:
#            e.ignore() 

    def findNum(self):
        pad = 25
        image_num = Image.open('test_crop.png')
        image1 = np.asarray(image_num)
        image1 = cv2.resize(image1,(256-2*pad,256-2*pad))
        image1 = cv2.copyMakeBorder(image1, pad, pad, pad, pad, cv2.BORDER_CONSTANT,value=(0,64,0))
        image1 = image1/255
        
        
        image1 = np.expand_dims(image1,axis=0)
        
        w0, h0 = image_num.size
        predict_corner = self.model_num.predict(image1)[0]
        predict_corner[0] = (predict_corner[0]-25)/(256-50)*w0 #- (25/256 * (w0+50))
        predict_corner[1] = (predict_corner[1]-25)/(256-50)*h0
        predict_corner[2] = (predict_corner[2]-25)/(256-50)*w0
        predict_corner[3] = (predict_corner[3]-25)/(256-50)*h0
        image2 = np.asarray(image_num)
        image2 = image2/255
        #image2= cv2.resize(image2,(image_num.width+pad//2,image_num.height+pad//2))
        #predict_corner = predict_corner
        ret = self.plot_corner(image2,predict_corner,color = (255,0,0))
        ret.save('crop_box.png')
    
    def dropEvent(self, e):
        p = e.mimeData().text().replace('file:///','')
        pixmap1 = QPixmap(p)
        self.imlabel1.setPixmap(pixmap1)
        self.imlabel1.setScaledContents(True)
        
        self.warpIDcard(p)
        pixmap2 = QPixmap('test_crop.png')
        self.imlabel2.setPixmap(pixmap2)
        self.imlabel2.setScaledContents(True)

        result = self.IDscore()
        self.textEdit.setText(str(result[0]))
    
        self.findNum()
        pixmap1 = QPixmap('crop_box.png')
        self.imlabel1.setPixmap(pixmap1)
        self.imlabel1.setScaledContents(True)
        
        pixmap3 = QPixmap('crop_numim.png')
#        ocr_img = Image.open('crop_numim.png')
#        num_text  = pytesseract.image_to_string(ocr_img, lang='eng')
        num_text,score = self.ocr_model()
        self.textEdit_ocr.setText(str(num_text))
        self.im_idnum.setPixmap(pixmap3)
        self.im_idnum.setScaledContents(True)
        print(num_text)
        print(score)
        
        print(num_text)
    
    def numid_check(self,numid):
        p1=0
        if(len(numid)==13):
            for i in range(0,len(numid)-1):
                p1 += (13-i)*int(numid[i])
            p2 = p1 % 11
            p3 = 11-p2
            p4 = p3%10
            
            if(p4==int(numid[12])):
                flag = True
            else:
                flag = False
        else:
            flag = False
        
        return flag
    
    def start_check(self):
        if(self.toggle_flag == False):
            self.start_webcam()
            self.toggle_flag = True
            
    def start_webcam(self):
        if(self.toggle_flag == False):
            self.capture = cv2.VideoCapture(0)
            self.capture.set(3,self.w)
            self.capture.set(4,self.h)
            
            self.timer=QTimer(self)
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(0)
    
    def update_frame(self):
        self.startButton.setDefault(False)
        self.startButton.setAutoDefault(False)
        ret,self.image=self.capture.read()
#        self.image=cv2.flip(self.image,1)
        self.displayImage(self.image,1)
    
    def ocr_model(self):
#        img = Image.open('crop_numim.png')
#        shape = img.size
#        img2 = img
#        draw = ImageDraw.Draw(img2)
#        stx=145/350 * shape[0]
#        sty=30/220 * shape[1]
#        edx=273/350 * shape[0]
#        edy= 50/220 * shape[1]
#        draw.rectangle([(stx,sty),(edx,edy)],outline = (255,0,0))
#        #print('sx = ',edx-stx)
#        #print('sy = ',edy-sty)
#        
        cropped = Image.open('crop_numim.png')
        shape = cropped.size
        scale = np.min((256 / (cropped.size[0]) , 64 / (cropped.size[1])) )
        resize = cropped.resize((int(shape[0]*scale),int(shape[1]*scale)))
        shape = resize.size
        delta_w = 256 - shape[0]
        delta_h = 64 - shape[1]
        padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
        new_im = ImageOps.expand(resize, padding, fill  = (255,255,255))
        new_im.save('newim.png')
        ocr_img = np.asarray(new_im)

        ocr_img=ocr_img/255
        
        ocr_img2 = np.expand_dims(ocr_img,axis = 0)

        result = self.model_ocr.predict(ocr_img2)
        #score = np.max(result, axis = 2)[0]
        #labels = np.argmax(result, axis = 2)[0]
        decode,score2 = self.map_code_to_alphabet(result)
        
        return decode , score2
    
    def map_code_to_alphabet(self,code):
        c,h,w = code.shape
        index = np.squeeze(code.argmax(axis = 2))
        out = ''
        confidency = np.squeeze(code.max(axis = 2))
        for i in range(len(index)):
            out=out + self.alphabet[index[i]]
        return out, confidency
        
        
    def pytes_ocr(self):
        img = Image.open('crop_numim.png')
        text = pytesseract.image_to_string(img,config="-c tessedit_char_whitelist=0123456789").split('\n')
        for line in range(len(text)):
            onlynum = re.sub("\D", "", text[line])
        return onlynum
    
        
    def stop_webcam(self):
#        pixmap1 = QPixmap('cam.png')
        
#        self.imlabel1.setPixmap(self.image)
#        self.imlabel1.setScaledContents(True)
        
        self.timer.stop()
        self.capture.release()
        self.capture = cv2.VideoCapture(0)
        self.capture.set(3,1920)
        self.capture.set(4,1080)
        
        self.timer=QTimer(self)
        ret,self.image=self.capture.read()
        self.timer.start(0)
        
        self.toggle_flag = False
#        
#        self.timer.stop()
        
        #img = Image.fromarray(self.image.astype('uint8'))
        #img.save('cam.png')
#        self.update_frame()
        cv2.imwrite('cam.png', self.image)
        
        self.warpIDcard('cam.png',mode = 'cam')
        
        pixmap2 = QPixmap('test_crop.png')
        result = self.IDscore()
        self.textEdit.setText(str(result[0][0]))
        self.imlabel2.setPixmap(pixmap2)
        self.imlabel2.setScaledContents(True)
        
        self.findNum()
       
        pixmap3 = QPixmap('crop_numim.png')
#        ocr_img = Image.open('crop_numim.png')
#        num_text  = pytesseract.image_to_string(ocr_img, lang='eng')
        num_text,score = self.ocr_model()
#        num_text = self.pytes_ocr()
        self.textEdit_ocr.setText(str(num_text))
        self.im_idnum.setPixmap(pixmap3)
        self.im_idnum.setScaledContents(True)
        print(num_text)
      
        real_id = self.numid_check(num_text)
        print(real_id)
        if(real_id):
            correctim = QPixmap('corrected.jpg')
           
            self.label_flag.setPixmap(correctim)
            self.label_flag.setScaledContents(True)
        else:
            wrongim = QPixmap('wrong.jpg')
            self.label_flag.setPixmap(wrongim)
            self.label_flag.setScaledContents(True)
        
        
#        print(score)
        
        
    def displayImage(self,img0,window=1):
        
        pil = Image.fromarray(img0)
        draw = ImageDraw.Draw(pil)
       
#        img = img[int(self.h/2 - self.crop_size[0]/2):int(self.h/2 + self.crop_size[0]/2) , int(self.w/2 - self.crop_size[1]/2):int(self.w/2 + self.crop_size[1]/2)]
        draw.rectangle([(  self.w/2 - self.crop_size[1]/2, self.h/2 - self.crop_size[0]/2),(  self.w/2 + self.crop_size[1]/2 , self.h/2 + self.crop_size[0]/2  )])
        img = np.array(pil)
        
        qformat=QImage.Format_Indexed8
        if len(img.shape)==3:
            if img.shape[2]==4:
                qformat=QImage.Format_RGBA8888
            else:
                qformat=QImage.Format_RGB888
        outImage=QImage(img,img.shape[1],img.shape[0],img.strides[0],qformat)
        outImage=outImage.rgbSwapped()
        
        if window==1:
            Pixmap = QPixmap.fromImage(outImage)
            self.imlabel1.setPixmap(Pixmap)
            self.imlabel1.setScaledContents(True)
            
if __name__ == '__main__':
    app=QApplication(sys.argv)
    window = WEBCAMUI()
    window.setWindowTitle('KengCWN PyQt5')
    window.show()
    sys.exit(app.exec_())

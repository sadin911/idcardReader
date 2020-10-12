# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 08:45:45 2019

@author: chonlatid.d
"""
import cv2
import tensorflow.python.keras
import numpy as np

from PIL import Image, ImageDraw, ImageOps
from tensorflow.python.keras.models import load_model

class ThaiIDreader:
    def __init__(self):
        self.model = load_model(r'kengshawguay.h5')
        self.score_model = load_model('zeroloss.h5')
        self.model_crop = load_model(r'127_model_4corner.h5')
        self.model_crop.load_weights('0_model_4corner2.h5.weights')
        self.model_ocr = load_model(r'num_ocr_inw.h5')
        self.model_date = load_model(r'date_ocr_05022019_with_contrast.h5')
        self.alphabet = u'0123456789'
        
    def plot_corner(self,input_pil,predict_corner,color = (255,0,0)):
        ret_img = input_pil
        try:
            crop_numim = ret_img.crop((int(predict_corner[0]),int(predict_corner[1]),int(predict_corner[2]),int(predict_corner[3])))
            draw = ImageDraw.Draw(ret_img)
            draw.rectangle([predict_corner[0],predict_corner[1],predict_corner[2],predict_corner[3] ],  outline=(255,0,0))
            return ret_img,crop_numim
        
        except Exception as e:
            print(str(e))

    def transformFourPoints(self,image_cv, pts):
        
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
        warped = cv2.warpPerspective(image_cv, M, (maxWidth, maxHeight))
    
        return warped
    
    def order_points(self,pts):
    	rect = np.zeros((4, 2), dtype="float32")
    	s = pts.sum(axis=1)
    	rect[0] = pts[np.argmin(s)]
    	rect[2] = pts[np.argmax(s)]
    	diff = np.diff(pts, axis=1)
    	rect[1] = pts[np.argmin(diff)]
    	rect[3] = pts[np.argmax(diff)]
   
    	return rect
    
    def warpIDcard(self,input_pil):

        image0 = input_pil
        image0 = image0.convert('RGB')
        
        image1 = np.asarray(image0)
        
        h,w,d= image1.shape
        crop_size = (0.4*h,0.4*w)
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
        
        return warped
    
    def findall(self):
        warped_pil = Image.open('test_crop.png')        
        image_crop = warped_pil.resize((256,256))
        imagecv = np.asarray(image_crop) 
        imagecv = imagecv/127.5 - 1
        imagecv = np.expand_dims(imagecv,axis=0)
        predict_rect = self.model_crop.predict(imagecv)[0]
        '''num,face,issue,expiry,birth,name_th'''
        ret_img = warped_pil
        
        ret_img,box_ploted = self.plot_corner_all(warped_pil,predict_rect[0:4],color = (255,0,0))
        ret_img_num = ret_img
        
        ret_img,box_ploted = self.plot_corner_all(warped_pil,predict_rect[4:8],color = (255,0,0))
        ret_img_face = ret_img

        ret_img,box_ploted = self.plot_corner_all(warped_pil,predict_rect[8:12],color = (255,0,0))
        ret_img_issue = ret_img

        ret_img,box_ploted = self.plot_corner_all(warped_pil,predict_rect[12:16],color = (255,0,0))
        ret_img_expiry = ret_img

        ret_img,box_ploted = self.plot_corner_all(warped_pil,predict_rect[16:20],color = (255,0,0))
        ret_img_birth = ret_img

        ret_img,box_ploted = self.plot_corner_all(warped_pil,predict_rect[20:24],color = (255,0,0))
        ret_img_name = ret_img

        return ret_img_num,ret_img_face,ret_img_issue,ret_img_expiry,ret_img_birth,ret_img_name
    
    def plot_corner_all(self,input_pil,predict_corner,color = (255,0,0)):
        crop_numim = input_pil
        ret_img = input_pil
        try:
            w = input_pil.size[0]
            h = input_pil.size[1]
            crop_numim = crop_numim.crop((int(predict_corner[0]/256*w-1), int(predict_corner[1]/256*h+1) , int(predict_corner[2]/256*w-1), int(predict_corner[3]/256*h+1)))
            crop_numim = Image.fromarray(crop_numim) 
            draw = ImageDraw.Draw(ret_img)
            draw.rectangle([predict_corner[0]/256*w-1,predict_corner[1]/256*h-1,predict_corner[2]/256*w-1,predict_corner[3]/256*h-1 ],  outline=(255,0,0))
            return crop_numim,ret_img
        
        except Exception as e:
            print(str(e))
            
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
    
    def ocr_model(self,input_pil):
        cropped = input_pil
        shape = cropped.size
        scale = np.min((256 / (cropped.size[0]) , 64 / (cropped.size[1])) )
        resize = cropped.resize((int(shape[0]*scale),int(shape[1]*scale)))
        shape = resize.size
        delta_w = 256 - shape[0]
        delta_h = 64 - shape[1]
        padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
        new_im = ImageOps.expand(resize, padding, fill  = (255,255,255))
        ocr_img = np.asarray(new_im)
        ocr_img=ocr_img/255
        ocr_img2 = np.expand_dims(ocr_img,axis = 0)
        try:
            result = self.model_ocr.predict(ocr_img2)
            decode,score2 = self.map_code_to_alphabet(result)
            return decode , score2
        
        except Exception as e:
            print(str(e))
            
    def date_model(self,path):
        cropped = Image.open(path)
        shape = cropped.size
        scale = np.min((256 / (cropped.size[0]) , 64 / (cropped.size[1])) )
        resize = cropped.resize((int(shape[0]*scale),int(shape[1]*scale)))
        shape = resize.size
        delta_w = 256 - shape[0]
        delta_h = 64 - shape[1]
        padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
        new_im = ImageOps.expand(resize, padding, fill  = (255,255,255))
        ocr_img = np.asarray(new_im)
        ocr_img = ocr_img/127.5-1
        
        ocr_img_en = np.expand_dims(ocr_img,axis = 0)
#        ocr_img_en = self.model_date_enc.predict(ocr_img_en)
#        ocr_img_en = self.model_date_enc_2.predict(ocr_img_en)
        ocr_img_en = np.squeeze(ocr_img_en)
        ocr_img_en = ((ocr_img_en+1)*127.5)/255
        ocr_img_en_pil = Image.fromarray((ocr_img_en*255).astype(np.uint8))
        ocr_img_en = np.expand_dims(ocr_img_en,axis = 0)
        result = self.model_date.predict(ocr_img_en)
        #score = np.max(result, axis = 2)[0]
        #labels = np.argmax(result, axis = 2)[0]
        decode,score2 = self.map_code_to_alphabet(result)
        
        return decode , score2 , ocr_img_en_pil
    
    def map_code_to_alphabet(self,code):
        c,h,w = code.shape
        index = np.squeeze(code.argmax(axis = 2))
        out = ''
        confidency = np.squeeze(code.max(axis = 2))
        for i in range(len(index)):
            out=out + self.alphabet[index[i]]
        return out, confidency
    
    def predictall(self,input_pil):
        self.warpIDcard()
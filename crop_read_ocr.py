# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 08:55:43 2019

@author: chonlatid.d
"""
from __future__ import print_function, division

import cv2
import re
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
import pytesseract
pytesseract.pytesseract.tesseract_cmd= r'Tesseract-OCR\tesseract.exe' 

#require function
def transformFourPoints(image, pts):
        
    	rect = order_points(pts)
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
    
def order_points(pts):

    	rect = np.zeros((4, 2), dtype="float32")
    
    	s = pts.sum(axis=1)
    	rect[0] = pts[np.argmin(s)]
    	rect[2] = pts[np.argmax(s)]
    
    
    	diff = np.diff(pts, axis=1)
    	rect[1] = pts[np.argmin(diff)]
    	rect[3] = pts[np.argmax(diff)]
   
    	return rect

def map_code_to_alphabet(code):
        alphabet = u'0123456789'
        c,h,w = code.shape
        index = np.squeeze(code.argmax(axis = 2))
        out = ''
        confidency = np.squeeze(code.max(axis = 2))
        for i in range(len(index)):
            out=out + alphabet[index[i]]
        return out, confidency

def ocr_model(pil_image):
        cropped = pil_image
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

        result = model_ocr.predict(ocr_img2)
        #score = np.max(result, axis = 2)[0]
        #labels = np.argmax(result, axis = 2)[0]
        decode,score2 = map_code_to_alphabet(result)
        
        return decode , score2

# initial image
img = Image.open(r'example\testcam3.jpg')
img = img.convert('RGB')
image = np.asarray(img) 
img_cv = np.asarray(img)

# warp model
model_warp = load_model(r'idcard_loc_model.h5')
imsize = 256 
image_warp_cv = cv2.resize(image,(imsize,imsize))
image_warp_cv = image_warp_cv / 127.5 - 1
image_warp_cv = np.expand_dims(image_warp_cv,axis=0)
warp_loc = model_warp.predict(image_warp_cv)
corner = np.reshape(warp_loc,(4,2))

ratio_h = img.height / imsize
ratio_w = img.width / imsize
corner[:,0] *= ratio_w  
corner[:,1] *= ratio_h

c = corner.astype(np.int)
c = np.reshape(c,(4,2))

warped_cv = transformFourPoints(img_cv,c)
warped_pil = Image.fromarray(warped_cv.astype('uint8'),'RGB')
warped_pil.save('test_crop.png')

# id card score model
model_score = load_model('ModelDisV3.h5')
image_score_cv = cv2.resize(warped_cv,(256,256))
image_score_cv = image_score_cv / 127.5 - 1.
image_score_cv = np.expand_dims(image_score_cv,axis=0)
score=model_score.predict(image_score_cv) #use model
print('id card score = ' + str(score[0,0]))

# numcrop from model
model_num = load_model(r'numcrop_model.h5')
pad = 25
image_num = Image.open('test_crop.png')
w0, h0 = image_num.size
image_num_cv = np.asarray(image_num)
image_num_cv = cv2.resize(image_num_cv,(256-2*pad,256-2*pad))
image_num_cv = cv2.copyMakeBorder(image_num_cv, pad, pad, pad, pad, cv2.BORDER_CONSTANT,value=(0,64,0))
image_num_cv = image_num_cv/255
image_num_cv = np.expand_dims(image_num_cv,axis=0)

predict_corner = model_num.predict(image_num_cv)[0]
predict_corner[0] = (predict_corner[0]-25)/(256-50)*w0 #- (25/256 * (w0+50))
predict_corner[1] = (predict_corner[1]-25)/(256-50)*h0
predict_corner[2] = (predict_corner[2]-25)/(256-50)*w0
predict_corner[3] = (predict_corner[3]-25)/(256-50)*h0

crop_numim_cv = np.asarray(image_num) 
crop_numim_cv = crop_numim_cv[int(predict_corner[1]):int(predict_corner[3]) , int(predict_corner[0]):int(predict_corner[2])]
crop_numim_pil = Image.fromarray(crop_numim_cv.astype('uint8'))

# ocr read
model_ocr = load_model(r'ocr_v3.h5')
image_ocr = crop_numim_pil
shape = image_ocr.size
scale = np.min((256 / (image_ocr.size[0]) , 64 / (image_ocr.size[1])) )
resize = image_ocr.resize((int(shape[0]*scale),int(shape[1]*scale)))
shape = resize.size
delta_w = 256 - shape[0]
delta_h = 64 - shape[1]

padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
new_im = ImageOps.expand(resize, padding, fill  = (255,255,255))

image_ocr_cv = np.asarray(new_im)
image_ocr_cv = image_ocr_cv/255
image_ocr_cv = np.expand_dims(image_ocr_cv,axis = 0)
result = model_ocr.predict(image_ocr_cv)
ocr_frommodel,score2 = map_code_to_alphabet(result)
print('ocr_from_model = '+ocr_frommodel)

# numcrop from pytesseract
image_tess = crop_numim_pil
text = pytesseract.image_to_string(image_tess,config="-c tessedit_char_whitelist=0123456789").split('\n')
for line in range(len(text)):
    onlynum = re.sub("\D", "", text[line])
ocr_fromtesseract = onlynum
print('ocr_from_tesseract = '+ocr_fromtesseract)

#output = score, ocr_frommodel, ocr_fromtesseract

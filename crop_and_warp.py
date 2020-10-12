# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 13:01:25 2018

@author: trainai
"""

import numpy as np
from keras.models import load_model
from PIL import Image
import glob
import os
import cv2
import imutils
from skimage.filters import threshold_local
import find_minbox
from skimage.morphology import reconstruction
from keras import backend as K

def order_points(pts):

	rect = np.zeros((4, 2), dtype="float32")

	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]


	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
   
	return rect

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

def relu6(x):
    return K.relu(x, max_value=6)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

model=load_model('0_mtinynet_drop2_slowlearn.h5') #load model

image0 = Image.open('S__10534934.jpg')

image1 = image0.resize((512,512))
image1 = np.asarray(image1)
image1 = image1 / 255 
image1 = np.expand_dims(image1,axis=0)
foreground = model.predict(image1)

foreground = model.predict(image1)
foreground = np.reshape(foreground,(512,512))
foreground = cv2.normalize(foreground,foreground,0,255,cv2.NORM_MINMAX)
foreground = cv2.cvtColor(foreground,cv2.COLOR_GRAY2RGB)
image = foreground
#foreground = 255-foreground
foreground[:,:,1] *= 0
foreground[:,:,2] *= 0
                    
                    
foreground = Image.fromarray(foreground.astype('uint8'))

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#image = cv2.equalizeHist(image.astype(np.uint8))
image = imutils.resize(image, height = 500)
    
gray = cv2.GaussianBlur(image, (1, 1), 0)
ret2,th2 = cv2.threshold(gray.astype(np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#                   
kernel_close = np.ones((10,10),np.uint8)
kernel_dila = np.ones((3,5),np.uint8)
ratio_h = image0.height / 500.0
ratio_w = image0.width / 500.0
image2 = np.asarray(image0)
orig = image2.copy()

seed = np.copy(th2)
seed[1:-1, 1:-1] = th2.max()
mask = th2

#filled = reconstruction(seed, mask, method='erosion')

closing = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel_close)
dilation = cv2.dilate(closing,kernel_dila,iterations = 1)
edged = cv2.Canny(dilation, 250,250)                  
im2, contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea)
hull = []
for i in range(len(contours)):
    hull.append(cv2.convexHull(contours[i], False))
image0 = np.asarray(image0)
image0 = cv2.normalize(image0,image1,0,255,cv2.NORM_MINMAX)
drawing = image0.astype(np.uint8)
drawing = np.squeeze(drawing)
p2 = []
for i in range(0,len(hull)):  
    p = hull[i]
    p[:,0,0] = (p[:,0,0]*ratio_w).astype('int') 
    p[:,0,1] = (p[:,0,1]*ratio_h).astype('int') 
    pt = p.reshape(p.shape[0],p.shape[2])
    p1 = find_minbox.minimum_bounding_rectangle(pt)
    p2.append(p1)
    
for i in range(0,1):
    color_contours = (0, 255, 0) # green - color for contours
    color = (255, 0, 0) # blue - color for convex hull
    # draw ith contour
#    contours[:][:,0,0] = (contours[:,0,0]*ratio_w).astype('int') 
#    contours[:][:,0,1] = (contours[:,0,1]*ratio_h).astype('int') 
#    cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
    # draw ith convex hull object
    p2 = (np.abs(p2)).astype(np.int)
    i = len(p2)-1
    cv2.drawContours(drawing, p2, -1, color, 1, 8) 
    
p2 = sorted(p2, key=cv2.contourArea)
screenCnt = p2[len(p2)-1]
warped = transformFourPoints(orig, screenCnt.reshape(4, 2))
warped = (warped).astype("uint8")
warped = Image.fromarray(warped.astype('uint8'),'RGB')
warped.save('test_crop.jpg')
#
drawing = Image.fromarray(drawing.astype('uint8'))
edged = Image.fromarray(edged.astype('uint8'))
dilation = Image.fromarray(dilation.astype('uint8'))
closing = Image.fromarray(closing.astype('uint8'))
#filled = Image.fromarray(filled.astype('uint8'))
gray = Image.fromarray(gray.astype('uint8'))
drawing.save('drawing.jpg')
print(str(flag))

import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

img = cv2.imread('image10.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
edged = cv2.Canny(bfilter, 140, 210) #Edge detection
kernel = np.ones((2,2), 'uint8')
edged = cv2.dilate(edged , kernel, iterations=1)
cv2.cvtColor(edged, cv2.COLOR_BGR2RGB)
fig = plt.figure()
fg1=fig.add_subplot(1,3,1)
plt.imshow(edged)
plt.show()


keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

try:
    # print(1)
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour,0.25*cv2.arcLength(contour,True), True)
        if len(approx) == 4:
            location = approx
            # print(contour)
            break
    print(location)
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0,255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)
    
    (x,y) = np.where(mask==255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    # cropped_image = gray[x1:x2+1, y1:y2+1]
    cropped_image = gray[x1-3:x2+3, y1+5:y2-2]
    
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)
    print(result)
    fg1=fig.add_subplot(1,3,2)
    plt.imshow(cropped_image)
    plt.show()

    text = result[0][-2]
except:
    # print(2)
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True), True)
        if len(approx) == 4:
            location = approx
            # print(contour)
            break
    print(location)
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0,255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)
    
    (x,y) = np.where(mask==255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    # cropped_image = gray[x1:x2+1, y1:y2+1]
    cropped_image = gray[x1-3:x2+3, y1+5:y2-2]
    
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)
    print(result)
    fg1=fig.add_subplot(1,3,2)
    plt.imshow(cropped_image)
    plt.show()

    text = result[0][-2]
font = cv2.FONT_HERSHEY_SIMPLEX
res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=0.5, color=(0,255,0), thickness=1, lineType=cv2.LINE_AA)
res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),2)
fg1=fig.add_subplot(1,3,3)
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
plt.show()
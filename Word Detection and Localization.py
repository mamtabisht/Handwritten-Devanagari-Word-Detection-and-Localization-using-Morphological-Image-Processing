# Text detection and localization 

# for CMATERgt 1.5.1 dataset
 
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import os
#%matplotlib inline 

from os import makedirs  
import glob
print(os.listdir("D:/1Dataset/Mixed _Script_Databases/CMATERgt1.5.1"))

#Read the train & test Images and preprocessing
for directory_path in glob.glob("D:/1Dataset/Mixed _Script_Databases/CMATERgt1.5.1"):
    #print(directory_path)
    for filename in glob.glob(os.path.join(directory_path, "*.bmp")):
        print(filename)
        img = cv2.imread(filename) 
        #grayscale
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #binary
        # applying different thresholding # techniques on the input image
        thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, 199, 5) 
        #plt.figure(figsize=(10,10))
        #plt.imshow( thresh1, cmap='gray', vmax=1, vmin=0)
        
        #remove salt & papper noise
        out = cv2.medianBlur( thresh1, 7)
        #plt.figure(figsize=(10,10))
        #plt.imshow(out)
        
        #dilation
        kernel = np.ones((15,25), np.uint8)
        #kernel = np.ones((25,25), np.uint8)
        img_dilation = cv2.dilate(out, kernel, iterations=1)
        #find contours
        ctrs, hier = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #sort contours
        sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
        #plt.figure(figsize=(15,15))
        current_axis = plt.gca()
        
        lst = []
        for i, ctr in enumerate(sorted_ctrs):
            sub_list = []
            # Get bounding box
            x, y, w, h = cv2.boundingRect(ctr)
            sub_list.append(y)
            sub_list.append(x)
            sub_list.append(w)
            sub_list.append(h)
            lst.append(sub_list)
            current_axis.add_patch(Rectangle((x, y), w, h, edgecolor = 'g', fill=False, linewidth=1)) 
            lst.sort()
        os.mkdir(filename[:-4])
        for i in range(len(lst)):
            #print(i)
            cv2.imwrite(filename[:-4] + '/'+ filename[:-4].split('\\')[-1]+'_'+ str(i) + '.jpg',img[lst[i][0]: lst[i][0] + lst[i][3], lst[i][1]: lst[i][1] + lst[i][2]])
      
#################################################################################
#for CALAM_new

from os import makedirs  
import glob
print(os.listdir("C:/Desktop/New folder/croped-HTRdata"))


imdir = 'C:/Desktop/New folder/croped-HTRdata/'
ext = ['png', 'jpg']    # Add image formats here

files = []
[files.extend(glob.glob(imdir + '*.' + e)) for e in ext]

#images = [cv2.imread(file) for file in files]

#Read the train & test Images and preprocessing
for filename in files :
    print(filename)
    img = cv2.imread(filename) 
    #grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #binary
    # applying different thresholding # techniques on the input image
    thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, 199, 5) 
    #plt.figure(figsize=(10,10))
    #plt.imshow( thresh1, cmap='gray', vmax=1, vmin=0)
    
    #remove salt & papper noise
    out = cv2.medianBlur( thresh1, 7)
    #plt.figure(figsize=(10,10))
    #plt.imshow(out)
    
    #dilation
    kernel = np.ones((15,25), np.uint8)
    #kernel = np.ones((25,25), np.uint8)
    img_dilation = cv2.dilate(out, kernel, iterations=1)
    #find contours
    #,ctrs, hier = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ctrs, hier = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    #plt.figure(figsize=(15,15))
    current_axis = plt.gca()
    
    lst = []
    for i, ctr in enumerate(sorted_ctrs):
        sub_list = []
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        sub_list.append(y)
        sub_list.append(x)
        sub_list.append(w)
        sub_list.append(h)
        lst.append(sub_list)
        current_axis.add_patch(Rectangle((x, y), w, h, edgecolor = 'g', fill=False, linewidth=1)) 
        lst.sort()
    os.mkdir(filename[:-4])
    for i in range(len(lst)):
        #print(i)
        cv2.imwrite(filename[:-4] + '/'+ filename[:-4].split('\\')[-1]+'_'+ str(i) + '.jpg',img[lst[i][0]: lst[i][0] + lst[i][3], lst[i][1]: lst[i][1] + lst[i][2]])
      
#################################################################################
#for PHDIndic_11
from os import makedirs  
import glob

print(os.listdir("D:/1Dataset/PhDIndic_11/devnagari"))

#Read the train & test Images and preprocessing

for directory_path in glob.glob("D:/1Dataset/PhDIndic_11/devnagari"):
    #print(directory_path)
    for filename in glob.glob(os.path.join(directory_path, "*.tif")):
        print(filename)
        img = cv2.imread(filename) 
        #grayscale
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #binary
        # applying different thresholding # techniques on the input image
        thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, 199, 5) 
        #plt.figure(figsize=(10,10))
        #plt.imshow( thresh1, cmap='gray', vmax=1, vmin=0)
        
        #remove salt & papper noise
        out = cv2.medianBlur( thresh1, 7)
        #plt.figure(figsize=(10,10))
        #plt.imshow(out)
        
        #dilation
        kernel = np.ones((15,25), np.uint8)
        #kernel = np.ones((25,25), np.uint8)
        img_dilation = cv2.dilate(out, kernel, iterations=1)
        #find contours
        #,ctrs, hier = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ctrs, hier = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        #sort contours
        sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
        #plt.figure(figsize=(15,15))
        current_axis = plt.gca()
        
        lst = []
        for i, ctr in enumerate(sorted_ctrs):
            sub_list = []
            # Get bounding box
            x, y, w, h = cv2.boundingRect(ctr)
            sub_list.append(y)
            sub_list.append(x)
            sub_list.append(w)
            sub_list.append(h)
            lst.append(sub_list)
            current_axis.add_patch(Rectangle((x, y), w, h, edgecolor = 'g', fill=False, linewidth=1)) 
            lst.sort()
        os.mkdir(filename[:-4])
        for i in range(len(lst)):
            #print(i)
            cv2.imwrite(filename[:-4] + '/'+ filename[:-4].split('\\')[-1]+'_'+ str(i) + '.tif',img[lst[i][0]: lst[i][0] + lst[i][3], lst[i][1]: lst[i][1] + lst[i][2]])
              
            
###################################################################################
# for single image

filename = 'D:/1Dataset/Mixed _Script_Databases/CMATERgt1.5.1/GTHE001.bmp'
#filename = 'C:/Desktop/New folder/croped-HTRdata/HIN_A_RA_0000.png'

#img-gray-adaptivethresholding-opening-CC-dialation-countour

# Load img
img = cv2.imread(filename)
print(type(img))
plt.figure(figsize=(10,10))
plt.imshow(img)
plt.show()   

#grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(10,10))
plt.imshow(gray)

#binary
# applying thresholding 
thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY_INV, 199, 5) 
plt.figure(figsize=(10,10))
plt.imshow( thresh1, cmap='gray', vmax=1, vmin=0)

########

out = cv2.medianBlur( thresh1, 7)
plt.figure(figsize=(10,10))
plt.imshow(out)

#Opening is just another name of erosion followed by dilation. It is useful in removing noise
kernel = np.ones((4,4), np.uint8)
opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
plt.figure(figsize=(10,10))
plt.imshow( opening, cmap='gray', vmax=1, vmin=0)
########

# do connected components processing
nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(opening, None, None, None, 8, cv2.CV_32S)

#get CC_STAT_AREA component as stats[label, COLUMN] 
areas = stats[1:,cv2.CC_STAT_AREA]

result = np.zeros((labels.shape), np.uint8)

for i in range(0, nlabels - 1):
    if areas[i] >= 100:   #keep
        result[labels == i + 1] = 255

plt.figure(figsize=(10,10))
plt.imshow(result, cmap='gray', vmax=1, vmin=0)

################

#dilation # it increases the white region in the image or size of foreground object increases
#kernel = np.ones((25,130), np.uint8)
kernel = np.ones((10,10), np.uint8)
kernel = np.ones((25,25), np.uint8)
img_dilation = cv2.dilate(out, kernel, iterations=1)
plt.figure(figsize=(10,10))
plt.imshow(img_dilation, cmap='gray', vmax=1, vmin=0)

################

#find contours
ctrs, hier = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#Contours is a Python list of all the contours in the image.
# Each individual contour is a Numpy array of (x,y) coordinates of boundary points of the object.

#sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

plt.figure(figsize=(15,15))
current_axis = plt.gca() # gets the current axes so that you can draw on it directly.

lst = []

for i, ctr in enumerate(sorted_ctrs):
   # print(ctr)
    sub_list = []
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)
    sub_list.append(y)
    sub_list.append(x)
    sub_list.append(w)
    sub_list.append(h)    
    lst.append(sub_list)
    current_axis.add_patch(Rectangle((x, y), w, h, edgecolor = 'g', fill=False, linewidth=1)) 

#print(img.shape)
plt.imshow(img)
#cv2.waitKey(0)
plt.show()

lst.sort()

os.mkdir(filename.split('/')[-1].split('.')[0])

for i in range(len(lst)):
    print(i)
    cv2.imwrite(filename.split('/')[-1].split('.')[0]+ '_'+ str(i) + '.jpg',img[lst[i][0]: lst[i][0] + lst[i][3], lst[i][1]: lst[i][1] + lst[i][2]])
      

#ROUGH
roi = img[54:190, 2:2180]
# show ROI
plt.figure(figsize=(10,10))
plt.imshow(roi) 

##################################################################################
#loop

from os import makedirs  
import glob
print(os.listdir("C:/Desktop/New folder/croped-HTRdata"))

#Read the train & test Images and preprocessing
 
for directory_path in glob.glob("C:/Desktop/New folder/croped-HTRdata"):
    #print(directory_path)
    for filename in glob.glob(os.path.join(directory_path, "*.png")):
        print(filename)
        img = cv2.imread(filename) 
        #grayscale
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #binary
        ret,thresh = cv2.threshold(gray,160,255,cv2.THRESH_BINARY_INV)
        #dilation
        kernel = np.ones((25,130), np.uint8)
        img_dilation = cv2.dilate(thresh, kernel, iterations=1)
        #find contours
        ctrs, hier = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        #sort contours
        sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
        #plt.figure(figsize=(15,15))
        current_axis = plt.gca()
        
        lst = []
        for i, ctr in enumerate(sorted_ctrs):
            sub_list = []
            # Get bounding box
            x, y, w, h = cv2.boundingRect(ctr)
            sub_list.append(y)
            sub_list.append(x)
            sub_list.append(w)
            sub_list.append(h)            
            lst.append(sub_list)
            current_axis.add_patch(Rectangle((x, y), w, h, edgecolor = 'g', fill=False, linewidth=1)) 
            lst.sort()
      
        for i in range(len(lst)):
            #print(i)
            cv2.imwrite(filename.split('\\')[0]+'/'+'seg_' +'/'+filename.split('\\')[-1][:-4]+'_'+ str(i) + '.jpg',img[lst[i][0]: lst[i][0] + lst[i][3], lst[i][1]: lst[i][1] + lst[i][2]])
   
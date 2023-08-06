The ArUco markers are fiducial square markers used for camera pose estimation. An ArUco marker is a synthetic square marker with an internal binary matrix enclosed within a wide black color border with a Unique Identifier.

![image](https://github.com/sud0x00/CVPR/assets/91898207/3768b4bc-3422-43c1-86c3-1a620b3039ea)


### Brush up of some basic concepts:


#### Erosion and Dilation 
During dilation, the image is expanded by adding the number of pixels to the boundaries of objects, whereas in erosion, pixels are removed from object boundaries. 

Dilation increases the object's size in the white shade, and the size of an object in black shades decreases. 

In Erosion, the regions of darker shades increase, and the white shade or brighter side decreases.


Dilation removes small white noises, and erosion helps join broken parts of an object. 

Dilation and erosion are performed on the binary images.


#### Contours

Contour refers to boundary pixels having the same color and intensity.



## Object Size Measuring Logic

1. Load the image, identify the objects whose size needs to be measured and grab their coordinates
2. Detect an ArUco marker in the image; calculate the pixel to inch conversion ratio with the known perimeter for the ArUco marker.
3. Compute the object size based on the object co-ordinates and the pixel to inch conversion


### Step 1 :

Load the image to find the contours detected for different shapes. 

To find the different shapes, convert the image to grayscale using cvtColor(), blur it slightly using GaussianBlur() to smoothen the image and remove any Gaussian noise from an image.

Find the edges in an image using the Canny() algorithm and then apply two morphological image processing operations dilate() and erode().

Find the edges of the objects using findContours().

Grab the contours to sort them from left to right.

### Step 2 : 

Detect an ArUco marker to calculate the pixel to inch conversion ratio

The image is analyzed to find square shapes that are candidates to be markers.

After the candidates are detected, validate their inner codification to ensure that they are ArUco markers.

Detect the ArUco marker, and find the marker's perimeter in the image using OpenCV’s arcLength() function.

Calculate the pixel to inch conversion ratio by dividing the perimeter of the ArUco marker in pixels by the perimeter of the ArUco marker in inches.

### Step 3 :

Compute the object size based on the object co-ordinates and the pixel to inch conversion

Loop over all the identified contours.

If the contour area is greater than the threshold, retrieve the bounding rectangle while considering the object's rotation using minAreaRect(). The minAreaRect() will return (x,y), (width, height), angle of rotation.

![image-1](https://github.com/sud0x00/CVPR/assets/91898207/5789ba72-5680-4e68-8468-d9300cb585d2)


You need four rectangle corners to draw the rectangle, which is obtained using boxPoints(). Order the points in the contour in top-left, top-right, bottom-right, and bottom-left, and then draw the outline of the rotated bounding box.

Centroid is calculated using the following formula

![image-2](https://github.com/sud0x00/CVPR/assets/91898207/f71d93c7-92c9-48d0-9fa8-21cdcaac6f0d)



Finally, calculate the width using the Euclidean distance between top-right and top-left.

Calculate the height using the Euclidean distance between bottom-left and top-left.

Divide height and width by pixel to inch conversion factor, and you have the dimension of your object in inches.


![image-3](https://github.com/sud0x00/CVPR/assets/91898207/c9cf5441-b908-4518-8fd6-268ad2a3f450)


## Complete Code : 
```
# import the necessary packages
from scipy.spatial import distance as dist
import numpy as np
import imutils
from imutils import contours
from imutils import perspective
import cv2

# detect aruco marker
def findArucoMarkers(img, markerSize = 6, totalMarkers=100, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(cv2.aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    #print(key)
    
    #Load the dictionary that was used to generate the markers.
    arucoDict = cv2.aruco.Dictionary_get(key)
    
    # Initialize the detector parameters using default values
    arucoParam = cv2.aruco.DetectorParameters_create()
    
    # Detect the markers
    bboxs, ids, rejected = cv2.aruco.detectMarkers(gray, arucoDict, parameters = arucoParam)
    return bboxs, ids, rejected
# find object size 

# Load image
image=cv2.imread("aruco_object.jpg")

# Resize image
image = imutils.resize(image, width=500)

# Convert BGR image to gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Remove Gaussian noise from the image
gray = cv2.GaussianBlur(gray, (7, 7), 0)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# sort the contours from left-to-right and initialize the
(cnts, _) = contours.sort_contours(cnts)

# for pixel to inch calibration 
pixelsPerMetric = None

# Detect Aruco marker and use 
#it's dimension to calculate the pixel to inch ratio
arucofound =findArucoMarkers(image, totalMarkers=100)
if  len(arucofound[0])!=0:
    print(arucofound[0][0][0])
    aruco_perimeter = cv2.arcLength(arucofound[0][0][0], True)
    print(aruco_perimeter)
    # Pixel to Inch ratio
    # perimeter of the aruco marker is 8 inches
    pixelsPerMetric = aruco_perimeter / 8
    print(" pixel to inch",pixelsPerMetric)
else:
    pixelsPerMetric=38.0

# loop over the contours individually
for c in cnts:
    
    # if the contour is not sufficiently large, ignore it
    if cv2.contourArea(c) < 2000:
        continue
    ''' bounding rectangle is drawn with minimum area, so it considers the rotation also. 
    The function used is cv.minAreaRect(). It returns a Box2D structure which contains following details - 
    ( center (x,y), (width, height), angle of rotation ). 
    But to draw this rectangle, we need 4 corners of the rectangle. 
    It is obtained by the function cv.boxPoints()
    '''      
    # compute the rotated bounding box of the contour
    box = cv2.minAreaRect(c)
    box = cv2.boxPoints(box)
    box = np.int0(box)
    cv2.drawContours(image,[box],0,(0,0,255),2)
    
    # Draw the centroid   
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.circle(image, (cX, cY), 5, (255, 255, 255), -1)
       
    # order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order, then draw the outline of the rotated bounding
    # box  
    (tl, tr, br, bl) = box
    width_1 = (dist.euclidean(tr, tl))
    height_1 = (dist.euclidean(bl, tl))
    d_wd= width_1/pixelsPerMetric
    d_ht= height_1/pixelsPerMetric
    
    #display the image with object width and height in inches
    cv2.putText(image, "{:.1f}in".format(d_wd),((int((tl[0]+ tr[0])*0.5)-15, int((tl[1] + tr[1])*0.5)-15)), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)
    cv2.putText(image, "{:.1f}in".format(d_ht),((int((tr[0]+ br[0])*0.5)+10, int((tr[1] + br[1])*0.5)+10)), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)
    # show the output image
    cv2.imshow("Image-dim", image)
    fname="size{}.jpg".format(str(i))
    cv2.imwrite(fname, image)
    key = cv2.waitKey(0)

cv2.destroyAllWindows()
```

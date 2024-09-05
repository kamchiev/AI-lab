import cv2
import numpy as np

image = []
img = cv2.imread("C:\Users\khayo\AI lab codes\Computer Vision\book.jpg")
image.append(img)
img = cv2.imread('doll.jpg')
image.append(img)
img = cv2.imread('calculator.jpg')
image.append(img)

#create the list containing the classes of the objects
classes = ['book','calculator','doll']

#create the descriptor database
def descriptor(images):
    descriptor_list = []
    orb = cv2.ORB_create(nfeatures = 1000)

    #extract the features 
    for image in images:
        kpt, des = orb.detectAndCompute(image,None)
        descriptor_list.append(des)
    
    return descriptor_list

# do the match 
def objClassification(frame,descriptor_list):
    orb = cv2.ORB_create(nfeatures=1000)
    kpt, des = orb.detectAndCompute(frame, None)

    # create the matcher 
    matcher = cv2.BFMatcher()
    best_matches = []

    # perform the matches with the database
    for descriptor in descriptor_list: 
        matches = matcher.knnMatch(des,descriptor,k=2)
        good = []

        for m,n in matches:
            if m.distance < n.distance * 0.8:
                good.append([m])
        best_matches.append(len(good))

    # classId
    classId = -1 
    if len(best_matches) > 0:
        max_val = max(best_matches)
        if max_val > 20: # we must have at least 10 matches
            classId = best_matches.index(max_val)
    return classId

#let's see 
descriptor_list = descriptor(image)
webcam = cv2.VideoCapture(0) # 0 is the index of the camera.

while True:
    # read the frame
    success, frame = webcam.read()
    obj_id = objClassification(frame,descriptor_list)

    if obj_id != -1:
        cv2.putText(frame,classes[obj_id],(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1,)
    
    cv2.imshow('Frame',frame)
    k = cv2.waitKey(30)
    if k == ord('q'):
        break


import cv2
import numpy as np

def onClick(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(dst_points) < 4:
            dst_points.append([x,y])
            cv2.circle(img_copy,(x,y),10,(0,0,255),-1)
            cv2.imshow('board img',img_copy)

base_img = cv2.imread("board.jpg")


img_copy = base_img.copy()

img = cv2.imread("taylor.jpg")

base_h, base_w = base_img.shape[:2]
img_h,img_w = img.shape[:2]

src_points = np.array([[0,0],[0,img_h],[img_w, img_h],[img_w,0]], dtype=np.float32)
dst_points = []


cv2.namedWindow('board',cv2.WINDOW_KEEPRATIO)
cv2.setMouseCallback('board',onClick)
cv2.imshow('board',base_img)
#cv2.imshow('img1',img)
cv2.waitKey(0)


#computing the homography matrix

dst_float = np.array(dst_points,dtype=np.float32)

H = cv2.getPerspectiveTransform(src_points,dst_float)

#apply H to the image to be warped

warped = cv2.warpPerspective(img,H,(base_w,base_h))
mask = np.zeros(base_img.shape,dtype=np.uint8)
cv2.fillConvexPoly(mask,np.int32(dst_points),(255,255,255))
#invert the mask
mask = cv2.bitwise_not(mask)

#apply the mask to billboard image
masked_billboard = cv2.bitwise_and(base_img,mask)

#apply the mask to warped image
final_img = cv2.bitwise_or(masked_billboard,warped)

cv2.namedWindow('taylor',cv2.WINDOW_KEEPRATIO)
cv2.imshow('taylor',final_img)
cv2.waitKey(0)

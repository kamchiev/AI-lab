import cv2
import numpy as np

# CARTOONIZED EFFECT
#img = cv2.imread('gerry.png')

cap = cv2.VideoCapture(0)

while True:
    img = cap.read()[1]



    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    img_gray = cv2.medianBlur(img_gray,5)

    # extract the edges with Laplacian
    edges = cv2.Laplacian(img_gray,cv2.CV_8U,ksize=5)

    #thresholding the edges -> for example if it is set to 100. everything below 100 gets black and everything over 100 gets white

    _, thresholded = cv2.threshold(edges, 70, 255,cv2.THRESH_BINARY_INV) #everything above 70 will be set to 255

    #get the colors with bileteral filter. it blurs inside the edges

    color_img = cv2.bilateralFilter(img,10,250,250)

    #merge color and edges
    skt = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
    sketch_img = cv2.bitwise_and(color_img,skt)

    cv2.namedWindow('Img',cv2.WINDOW_KEEPRATIO)
    cv2.imshow("img",sketch_img)
    k = cv2.waitKey(1)

    if k == ord('q'):
        break

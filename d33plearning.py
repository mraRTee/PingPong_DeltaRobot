################################################################################## ez itt a jó!!!!!!!!!!!!!!
import cv2
import numpy as np
import matplotlib.pyplot as plt

listx_felso = []
listy_felso = []
listx_also = []
listy_also = []
list_kiirx_f = []
list_kiiry_f = []
list_kiirx_l = []
list_kiiry_l = []


cap1 = cv2.VideoCapture("testvid10f.mp4")
cap2 = cv2.VideoCapture("testvid10l.mp4")
# cap1.set(cv2.CAP_PROP_FPS, 60) #60 fpsre állítás
# cap2.set(cv2.CAP_PROP_FPS, 60)
x_a = 0
y_a = 0
x_b = 0
y_b = 0
kd1 = ""
kd2 = ""
s_1 = False
s_2 = False
cn_1 = 0
cn_2 = 0
cim1 = ""
txt = ".txt"
e = 0
g = 0

def plot():
    # AZ EGÉSZET PLOTTOLJA KI MOST!!!!!!!!!!!!!!!!!!!!!
    plt.plot(list_kiiry_f, list_kiirx_f, 'ro')
    plt.axis([0, 1280, 720, 0])
    plt.plot(list_kiiry_l, list_kiirx_l, 'go')
    plt.axis([0, 1280, 720, 0])
    plt.show()


while 1:  # folyamatosan veszi a jelet
    _, vid1 = cap1.read()  # minden kockát olvas
    _, vid2 = cap2.read()  # minden kockát olvas
    try:
        hsv1 = cv2.cvtColor(vid1, cv2.COLOR_BGR2HSV)  # ez itt a konverzió
        hsv2 = cv2.cvtColor(vid2, cv2.COLOR_BGR2HSV)  # ez itt a konverzió
    except:
        with open("x_felső_full.txt", 'w+') as f:
            for item in list_kiirx_f:
                f.write("%s\n" % item)
        with open("y_felső_full.txt", 'w+') as f:
            for item in list_kiiry_f:
                f.write("%s\n" % item)
        plot()
    lower_orange1 = np.array([0, 110, 110])
    upper_orange1 = np.array([40, 255, 255]) #új videóknál ez az atom!!!
    lower_orange2 = np.array([0, 130, 130])
    upper_orange2 = np.array([70, 255, 160])

    mask1 = cv2.inRange(hsv1, lower_orange1, upper_orange1)  # ez maszkolja ki
    mask2 = cv2.inRange(hsv2, lower_orange2, upper_orange2)  # ez maszkolja ki
    # res1 = cv2.bitwise_and(vid1, vid1, mask=mask1)  # a kimaszkolt és igazi kép összeÉSelve.

    external_poly_1_a = np.array([[[0, 0], [190, 0], [199, 720], [0, 720]]], dtype=np.int32)
    external_poly_2_a = np.array([[[0, 0], [1280, 0], [1280, 3], [0, 15]]], dtype=np.int32)
    external_poly_3_a = np.array([[[971, 0], [1280, 0], [1280, 720], [985, 720]]], dtype=np.int32)
    external_poly_4_a = np.array([[[0, 570], [1280, 550], [1280, 720], [0, 720]]], dtype=np.int32)

    external_poly_1_b = np.array([[[0, 0], [150, 80], [150, 720], [0, 720]]], dtype=np.int32)
    external_poly_2_b = np.array([[[0, 0], [1280, 0], [1280, 250], [0, 250]]], dtype=np.int32)
    external_poly_3_b = np.array([[[1100, 0], [1280, 0], [1280, 720], [1100, 720]]], dtype=np.int32)
    external_poly_4_b = np.array([[[0, 650], [1280, 650], [1280, 720], [0, 720]]], dtype=np.int32)
    # [[[10, 10], [100, 10], [100, 100], [10, 100]]] kis négyzet
    # https: // stackoverflow.com / questions / 56813343 / masking - out - a - specific - region - in -opencv - python


    # itt van valami kurva nagy gáZ!!!!!!!!!!
    cv2.fillPoly(mask1, external_poly_1_a, (0, 0, 0))
    cv2.fillPoly(mask1, external_poly_2_a, (0, 0, 0))
    cv2.fillPoly(mask1, external_poly_3_a, (0, 0, 0))
    cv2.fillPoly(mask1, external_poly_4_a, (0, 0, 0))
    #########################################################
    cv2.fillPoly(mask2, external_poly_1_b, (0, 0, 0))
    cv2.fillPoly(mask2, external_poly_2_b, (0, 0, 0))
    cv2.fillPoly(mask2, external_poly_3_b, (0, 0, 0))
    cv2.fillPoly(mask2, external_poly_4_b, (0, 0, 0))

    cv2.fillPoly(mask2, external_poly_1_b, (0, 0, 0))

    # ez a rész csak a videós megjelenítéshez kell
    # cv2.fillPoly(vid1, external_poly_1_a, (0, 0, 0))
    # cv2.fillPoly(vid1, external_poly_2_a, (0, 0, 0))
    # cv2.fillPoly(vid1, external_poly_3_a, (0, 0, 0))
    # cv2.fillPoly(vid1, external_poly_4_a, (0, 0, 0))

    cv2.fillPoly(vid2, external_poly_1_b, (0, 0, 0))
    cv2.fillPoly(vid2, external_poly_2_b, (0, 0, 0))
    cv2.fillPoly(vid2, external_poly_3_b, (0, 0, 0))
    cv2.fillPoly(vid2, external_poly_4_b, (0, 0, 0))

    cv2.line(vid1, (int(y_a), int(x_a)), (int(y_a), int(x_a)), (0, 0, 255), 15)
    cv2.line(vid2, (int(y_b), int(x_b)), (int(y_b), int(x_b)), (0, 255, 0), 15)

    x_a = 0
    y_a = 0
    x_b = 0
    y_b = 0

    cv2.imshow('vid1', vid1)  # mit mutat
    cv2.imshow('vid2', vid2)
    cv2.waitKey(1)  # k = cv2.waitKey(1) & 0xFF #itt ez nem tudom mi volt, de ugyanúgy működik így

    npImg1 = np.asarray(mask1)  #melyikről vegye a mintát
    npImg2 = np.asarray(mask2)
    kd_b = np.argwhere(npImg2 == 255)
    kd_a = np.argwhere(npImg1 == 255)
    kd_a = str(kd_a)
    kd_b = str(kd_b)
    for i in range(15):
        # ha nincs külön minden if-re egy try akkor csak akkor fog működni, ha a fenti kamera detektál,
        # lehet az még hasznos lehet
        try:
            if kd_a[i] == ' ' and kd1 and not x_a:
                x_a = kd1
                x_a = int(x_a)
                kd1 = ""
        except:
            pass
        try:
            if kd_a[i] == ' ' and kd1 and x_a:
                y_a = kd1
                y_a = int(y_a)
                kd1 = ""
        except:
            pass
        try:
            if kd_a[i] == '0' or kd_a[i] == '1' or kd_a[i] == '2' or kd_a[i] == '3' or kd_a[i] == '4' or kd_a[i] == '5'\
                    or kd_a[i] == '6' or kd_a[i] == '7' or kd_a[i] == '8' or kd_a[i] == '9':
                kd1 += kd_a[i]
        except:
            pass
        try:
            if kd_b[i] == ' ' and kd2 and not x_b:
                x_b = kd2
                x_b = int(x_b)
                kd2 = ""
        except:
            pass
        try:
            if kd_b[i] == ' ' and kd2 and x_b:
                y_b = kd2
                y_b = int(y_b)
                kd2 = ""
        except:
            pass
        try:
            if kd_b[i] == '0' or kd_b[i] == '1' or kd_b[i] == '2' or kd_b[i] == '3' or kd_b[i] == '4' or kd_b[i] == '5'\
                    or kd_b[i] == '6' or kd_b[i] == '7' or kd_b[i] == '8' or kd_b[i] == '9':
                kd2 += kd_b[i]
        except:
            pass
    kd1 = ""
    kd2 = ""
    if x_a != 0 and y_a != 0 and s_1 is False:
        listx_felso.append(int(x_a))
        listy_felso.append(int(y_a))

        if cn_1 > 2:
            try:
                if listy_felso[cn_1] > listy_felso[cn_1 - 1]:
                    del listy_felso[-1]
                    del listx_felso[-1]
                    cim1 = str(e)+ "x_felso" + txt
                    with open(cim1, 'w+') as f:
                        for item in listx_felso:
                            f.write("%s\n" % item)
                    cim1 = str(e) + "y_felso" + txt
                    with open(cim1, 'w+') as f:
                        for item in listy_felso:
                            f.write("%s\n" % item)
                    e += 1
                    s_1 = True
            except:
                cn_1 -= 1
        cn_1 += 1

    if x_a == 0 and y_a == 0 and s_1:
        print("x felső: ", listx_felso)
        print("y felső: ", listy_felso)
        list_kiirx_f+=listx_felso
        list_kiiry_f+=listy_felso
        listx_felso = []
        listy_felso = []
        s_1 = False

    # if x_b != 0 and y_b != 0 and s_2 is False:
    #     listx_also.append(int(x_b))
    #     listy_also.append(int(y_b))
    #     # list_kiirx_l.append(int(x_b))
    #     # list_kiiry_l.append(int(y_b))
    #
    #     if cn_2 > 2:
    #         try:
    #             if listy_also[cn_2] > listy_also[cn_2 - 1]:
    #                 del listy_also[-1]
    #                 del listx_also[-1]
    #                 cim1 = str(g)+ "x_also" + txt
    #                 with open(cim1, 'w+') as f:
    #                     for item in listx_felso:
    #                         f.write("%s\n" % item)
    #                 cim1 = str(g) + "y_also" + txt
    #                 with open(cim1, 'w+') as f:
    #                     for item in listy_felso:
    #                         f.write("%s\n" % item)
    #                 g += 1
    #                 s_2 = True
    #         except:
    #             cn_2 -= 1
    #     cn_2 += 1
    # if x_b == 0 and y_b == 0 and s_2:
    #     print("x alsó: ", listx_also)
    #     print("y alsó: ", listy_also)
    #     list_kiirx_l+=listx_also
    #     list_kiiry_l+=listy_also
    #     listx_also = []
    #     listy_also = []
    #     s_2 = False


cv2.destroyAllWindows()

# import numpy as np
# import cv2
#
# cap = cv2.VideoCapture('testvid.mp4')
# cap.set(cv2.CAP_PROP_FPS, 30)
# x=0
# y=0
#
# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     cv2.line(frame, (x, x), (y, y), (0, 255, 100), 5)
#     # Our operations on the frame come here
#     #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Display the resulting frame
#     cv2.imshow('frame',frame)
#     x+=1
#     y+=1
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()


# import cv2 as cv
# import sys
# img = cv.imread("testkep.png")
# if img is None:
#     sys.exit("Could not read the image.")
#
# pixel= img[200, 550]
# xk=0
# yk=0
# zk=0
# piros= ""
# zold=""
# kek=""
# seged2=0
# seged3=0
# szokoz=0
# counter1=0
# for xk in range(1280):
#     for yk in range(720):
#         szin=img[yk,xk]
#         szin=str(szin)
#         for zk in szin:
#             if zk=="[":
#                 continue
#             if zk=="]":
#                 piros = int(piros)
#                 zold = int(zold)
#                 kek = int(kek)
#                 if piros >0 and zold >100 and kek>100 and piros <70 and zold <255 and kek <255:
#                     img = cv.circle(img, (xk,yk), 2, (0, 0, 255), -1)
#                     cv.imshow("Display window", img)
#                 seged3+=1
#                 piros = ""
#                 zold = ""
#                 kek = ""
#                 seged2=0
#                 break
#             if seged2==0 and zk!=" ":
#                 piros+=zk
#             if seged2==0 and zk==" ":
#                 if(piros==""):
#                     continue
#                 else:
#                     seged2+=1
#                     continue
#             if seged2==1 and zk!=" ":
#                 zold+=zk
#             if seged2==1 and zk==" ":
#                 if (zold == ""):
#                     continue
#                 else:
#                     seged2 += 1
#                     continue
#             if seged2==2 and zk!=" ":
#                 kek+=zk
#             if seged2==2 and zk==" ":
#                 if (kek == ""):
#                     continue
#                 else:
#                     seged2 += 1
#                     continue
#
#
# k = cv.waitKey(0)
# if k == ord("s"):
#     cv.imwrite("starry_night.png", img)
#
# import cv2 as cv
# import sys
# img = cv.imread("testkep.png")
# xk=0
# yk=0
# zk=0
# piros= ""
# zold=""
# kek=""
# seged2=0
# seged3=0
# szokoz=0
# counter1=0
# for xk in range(1280):
#     for yk in range(720):
#         szin=img[yk,xk]
#         szin=str(szin)
#         for zk in szin:
#             if zk=="[":
#                 continue
#             if zk=="]":
#                 piros = int(piros)
#                 zold = int(zold)
#                 kek = int(kek)
#                 if piros >0 and zold >100 and kek>100 and piros <70 and zold <255 and kek <255:
#                     img = cv.circle(img, (xk,yk), 2, (0, 0, 255), -1)
#                     cv.imshow("Display window", img)
#                 seged3+=1
#                 piros = ""
#                 zold = ""
#                 kek = ""
#                 seged2=0
#                 break
#             if seged2==0 and zk!=" ":
#                 piros+=zk
#             if seged2==0 and zk==" ":
#                 if(piros==""):
#                     continue
#                 else:
#                     seged2+=1
#                     continue
#             if seged2==1 and zk!=" ":
#                 zold+=zk
#             if seged2==1 and zk==" ":
#                 if (zold == ""):
#                     continue
#                 else:
#                     seged2 += 1
#                     continue
#             if seged2==2 and zk!=" ":
#                 kek+=zk
#             if seged2==2 and zk==" ":
#                 if (kek == ""):
#                     continue
#                 else:
#                     seged2 += 1
#                     continue
#
#
# k = cv.waitKey(0)
# if k == ord("s"):
#     cv.imwrite("starry_night.png", img)
#
# lower_orange = np.array([2,150,150])  # narancsárga megtalálása
# upper_orange = np.array([40,255,255])
#
# import cv2
# import numpy as np
#
# font = cv2.FONT_HERSHEY_SIMPLEX
#
#
# # mouse callback function
# def draw_circle(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         i = 0
#         while True:
#             cv2.imshow('image', img)  # to display the characters
#             k = cv2.waitKey(0)
#             cv2.putText(img, chr(k), (x + i, y), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
#             i += 10
#             # Press q to stop writing
#             if k == ord('q'):
#                 break
#
#
# # Create a black image, a window and bind the function to window
# img = np.zeros((512, 512, 3), np.uint8)
# cv2.namedWindow('image')
# cv2.setMouseCallback('image', draw_circle)
#
# while True:
#     cv2.imshow('image', img)
#     if cv2.waitKey(20) == 27:
#         break
# cv2.destroyAllWindows()
#
# import cv2
# import numpy as np
#
# image = cv2.imread("testkep.png")
#
# # Create a named colour
# hsv1 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# lower_orange = np.array([0,120,120])  # narancsárga megtalálása
# upper_orange = np.array([50,255,255])
# mask1 = cv2.inRange(hsv1, lower_orange, upper_orange) #ez maszkolja ki
# res1 = cv2.bitwise_and(image, image, mask=mask1)  # a kimaszkolt és igazi kép összeÉSelve.
#
#
# cv2.imshow('frame',res1)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# import numpy as np
# import cv2
#
# cap = cv2.VideoCapture(1)
#
# while(True):
#     # Capture frame-by-frame
#     _, frame = cap.read()
#
#     # Our operations on the frame come here
#     gray = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
#
#     # Display the resulting frame
#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()
#
#
# import numpy as np
# import cv2 as cv
# import glob
# # termination criteria
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# objp = np.zeros((6*9,3), np.float32)
# objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
# # Arrays to store object points and image points from all the images.
# objpoints = [] # 3d point in real world space
# imgpoints = [] # 2d points in image plane.
# images = glob.glob('*.jpg')
# for fname in images:
#     img = cv.imread(fname)
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     # Find the chess board corners
#     ret, corners = cv.findChessboardCorners(gray, (9,6), None)
#     # If found, add object points, image points (after refining them)
#     if ret == True:
#         objpoints.append(objp)
#         corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
#         imgpoints.append(corners)
#         # Draw and display the corners
#         cv.drawChessboardCorners(img, (9,6), corners2, ret)
#         cv.imshow('img', img)
#         cv.waitKey(500)
# cv.destroyAllWindows()
# ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# img = cv.imread('2.jpg')
# h,  w = img.shape[:2]
# newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
# # undistort
# dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# # crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# cv.imwrite('calibresult.png', dst)
#
#
#
# import numpy as np
# import cv2 as cv
# import glob
# # termination criteria
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# objp = np.zeros((6*9,3), np.float32)
# objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
# # Arrays to store object points and image points from all the images.
# objpoints = [] # 3d point in real world space
# imgpoints = [] # 2d points in image plane.
# images = glob.glob('*.jpg')
# for fname in images:
#     img = cv.imread(fname)
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     # Find the chess board corners
#     ret, corners = cv.findChessboardCorners(gray, (9,6), None)
#     # If found, add object points, image points (after refining them)
#     if ret == True:
#         objpoints.append(objp)
#         corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
#         imgpoints.append(corners)
#         # Draw and display the corners
#         cv.drawChessboardCorners(img, (9,6), corners2, ret)
#         cv.imshow('img', img)
#         cv.waitKey(500)
# cv.destroyAllWindows()
# ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# img = cv.imread('2.jpg')
# h,  w = img.shape[:2]
# newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
# # undistort
# dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# # crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# cv.imwrite('calibresult.png', dst)
#
#
#
# import cv2
# import numpy as np
#
# cap = cv2.VideoCapture('asd.mp4')
#
# while(1):
#
#     # Take each frame
#     _,frame = cap.read()
#
#     # Convert BGR to HSV
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     # define range of blue color in HSV
#     lower_blue = np.array([10,0,0])
#     upper_blue = np.array([25,255,255])
#
#     # Threshold the HSV image to get only blue colors
#     mask = cv2.inRange(hsv, lower_blue, upper_blue)
#
#     # Bitwise-AND mask and original image
#     res = cv2.bitwise_and(frame,frame, mask= mask)
#
#     cv2.imshow('frame',frame)
#     cv2.imshow('mask',mask)
#     cv2.imshow('res',res)
#     k = cv2.waitKey(5) & 0xFF
#     if k == 27:
#         break
# cv2.destroyAllWindows()
#
# cap = cv2.VideoCapture('teset1.mp4')
#
#
# import numpy as np
# import cv2
#
# cap = cv2.VideoCapture(0)
#
# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#
#     # Our operations on the frame come here
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Display the resulting frame
#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()
#
#
# import numpy as np
# import cv2
#
# img = cv2.imread('asd.png',1)
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # USAGE
# # python ball_tracking.py --video ball_tracking_example.mp4
# # python ball_tracking.py
#
# # import the necessary packages
# from collections import deque
# from imutils.video import VideoStream
# import numpy as np
# import argparse
# import cv2
# import imutils
# import time
#
# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-v", "--video", default="teset1.mp4",help="path to the (optional) video file")
# ap.add_argument("-b", "--buffer", type=int, default=64,help="max buffer size")
# args = vars(ap.parse_args())
#
# # define the lower and upper boundaries of the "green"
# # ball in the HSV color space, then initialize the
# # list of tracked points
# greenLower = (60, 100, 100)
# greenUpper = (19, 73, 93)
# pts = deque(maxlen=args["buffer"])
#
# # if a video path was not supplied, grab the reference
# # to the webcam
# if not args.get("video", False):
# 	vs = VideoStream(src=0).start()
#
# # otherwise, grab a reference to the video file
# else:
# 	vs = cv2.VideoCapture(args["video"])
#
# # allow the camera or video file to warm up
# time.sleep(2.0)
#
# # keep looping
# while True:
# 	# grab the current frame
# 	frame = vs.read()
#
# 	# handle the frame from VideoCapture or VideoStream
# 	frame = frame[1] if args.get("video", False) else frame
#
# 	# if we are viewing a video and we did not grab a frame,
# 	# then we have reached the end of the video
# 	if frame is None:
# 		break
#
# 	# resize the frame, blur it, and convert it to the HSV
# 	# color space
# 	frame = imutils.resize(frame, width=600)
# 	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
# 	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
#
# 	# construct a mask for the color "green", then perform
# 	# a series of dilations and erosions to remove any small
# 	# blobs left in the mask
# 	mask = cv2.inRange(hsv, greenLower, greenUpper)
# 	mask = cv2.erode(mask, None, iterations=2)
# 	mask = cv2.dilate(mask, None, iterations=2)
#
# 	# find contours in the mask and initialize the current
# 	# (x, y) center of the ball
# 	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# 	cnts = imutils.grab_contours(cnts)
# 	center = None
#
# 	# only proceed if at least one contour was found
# 	if len(cnts) > 0:
# 		# find the largest contour in the mask, then use
# 		# it to compute the minimum enclosing circle and
# 		# centroid
# 		c = max(cnts, key=cv2.contourArea)
# 		((x, y), radius) = cv2.minEnclosingCircle(c)
# 		M = cv2.moments(c)
# 		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
#
# 		# only proceed if the radius meets a minimum size
# 		if radius > 10:
# 			# draw the circle and centroid on the frame,
# 			# then update the list of tracked points
# 			cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
# 			cv2.circle(frame, center, 5, (0, 0, 255), -1)
#
# 	# update the points queue
# 	pts.appendleft(center)
#
# 	# loop over the set of tracked points
# 	for i in range(1, len(pts)):
# 		# if either of the tracked points are None, ignore
# 		# them
# 		if pts[i - 1] is None or pts[i] is None:
# 			continue
#
# 		# otherwise, compute the thickness of the line and
# 		# draw the connecting lines
# 		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
# 		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
#
# 	# show the frame to our screen
# 	cv2.imshow("Frame", frame)
# 	key = cv2.waitKey(1) & 0xFF
#
# 	# if the 'q' key is pressed, stop the loop
# 	if key == ord("q"):
# 		break
#
# # if we are not using a video file, stop the camera video stream
# if not args.get("video", False):
# 	vs.stop()
#
# # otherwise, release the camera
# else:
# 	vs.release()
#
# # close all windows
# cv2.destroyAllWindows()

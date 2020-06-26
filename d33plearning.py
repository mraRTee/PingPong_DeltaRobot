################################################################################## ez itt a jó!!!!!!!!!!!!!!
import cv2
import numpy as np
import matplotlib.pyplot as plt

listx_1 = []
listy_1 = []
listx_2 = []
listy_2 = []

cap1 = cv2.VideoCapture("testvid8_mrr.mp4")
cap2 = cv2.VideoCapture("testvid7.mp4")
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
cim1=""
cim2=""
txt=".txt"

def szoveg_fajl():
    for e in range(4):
        cim1=str(e)+txt
        with open(cim1, 'a+') as f:
            for item in listx_1:
                f.write("%s\n" % item)
        with open('y.txt', 'a+') as f:
            for item in listy_1:
                f.write("%s\n" % item)


def plot():
    plt.plot(listy_1, listx_1, 'ro')
    plt.axis([0, 1280, 720, 0])
    plt.plot(listy_2, listx_2, 'go')
    plt.axis([0, 1280, 720, 0])
    plt.show()


while 1:  # folyamatosan veszi a jelet
    _, vid1 = cap1.read()  # minden kockát olvas
    _, vid2 = cap2.read()  # minden kockát olvas
    try:
        hsv1 = cv2.cvtColor(vid1, cv2.COLOR_BGR2HSV)  # ez itt a konverzió
        hsv2 = cv2.cvtColor(vid2, cv2.COLOR_BGR2HSV)  # ez itt a konverzió
    except:
        # print(listx_1)
        # print(listy_1)
        szoveg_fajl()
        plot()
    lower_orange1 = np.array([0, 110, 110])
    upper_orange1 = np.array([40, 255, 255]) #új videóknál ez az atom!!!
    lower_orange2 = np.array([25, 110, 150])
    upper_orange2 = np.array([40, 255, 255]) #új videóknál ez az atom!!!

    mask1 = cv2.inRange(hsv1, lower_orange1, upper_orange1)  # ez maszkolja ki
    mask2 = cv2.inRange(hsv2, lower_orange2, upper_orange2)  # ez maszkolja ki
    # res1 = cv2.bitwise_and(vid1, vid1, mask=mask1)  # a kimaszkolt és igazi kép összeÉSelve.

    external_poly_1_a = np.array([[[0, 0], [325, 0], [350, 720], [0, 720]]], dtype=np.int32)
    external_poly_2_a = np.array([[[0, 0], [1280, 0], [1280, 100], [0, 130]]], dtype=np.int32)
    external_poly_3_a = np.array([[[1035, 0], [1280, 0], [1280, 720], [1045, 720]]], dtype=np.int32)
    external_poly_4_a = np.array([[[0, 600], [1280, 550], [1280, 720], [0, 720]]], dtype=np.int32)

    external_poly_1_b = np.array([[[0, 0], [50, 50], [50, 720], [0, 720]]], dtype=np.int32)
    external_poly_2_b = np.array([[[0, 0], [1280, 0], [1280, 150], [0, 150]]], dtype=np.int32)
    external_poly_3_b = np.array([[[1100, 0], [1280, 0], [1280, 720], [1100, 720]]], dtype=np.int32)
    external_poly_4_b = np.array([[[0, 450], [1280, 450], [1280, 720], [0, 720]]], dtype=np.int32)
    # [[[10, 10], [100, 10], [100, 100], [10, 100]]] kis négyzet
    # https: // stackoverflow.com / questions / 56813343 / masking - out - a - specific - region - in -opencv - python

    cv2.fillPoly(mask1, external_poly_1_a, (0, 0, 0))
    cv2.fillPoly(mask1, external_poly_2_a, (0, 0, 0))
    cv2.fillPoly(mask1, external_poly_3_a, (0, 0, 0))
    cv2.fillPoly(mask1, external_poly_4_a, (0, 0, 0))

    cv2.fillPoly(mask2, external_poly_1_b, (0, 0, 0))
    cv2.fillPoly(mask2, external_poly_2_b, (0, 0, 0))
    cv2.fillPoly(mask2, external_poly_3_b, (0, 0, 0))
    cv2.fillPoly(mask2, external_poly_4_b, (0, 0, 0))

    # cv2.fillPoly(vid1, external_poly_1_a, (0, 0, 0))
    # cv2.fillPoly(vid1, external_poly_2_a, (0, 0, 0))
    # cv2.fillPoly(vid1, external_poly_3_a, (0, 0, 0))
    # cv2.fillPoly(vid1, external_poly_4_a, (0, 0, 0))
    #
    # cv2.fillPoly(vid2, external_poly_1_b, (0, 0, 0))
    # cv2.fillPoly(vid2, external_poly_2_b, (0, 0, 0))
    # cv2.fillPoly(vid2, external_poly_3_b, (0, 0, 0))
    # cv2.fillPoly(vid2, external_poly_4_b, (0, 0, 0))
    #
    # cv2.line(vid1, (int(y_a), int(x_a)), (int(y_a), int(x_a)), (0, 0, 255), 15)
    # cv2.line(vid2, (int(y_b), int(x_b)), (int(y_b), int(x_b)), (0, 255, 0), 15)
    x_a = 0
    y_a = 0
    x_b = 0
    y_b = 0
    # cv2.imshow('vid1', vid1)  # mit mutat
    # cv2.imshow('vid2', vid2)
    cv2.waitKey(1)  # k = cv2.waitKey(1) & 0xFF #itt ez nem tudom mi volt, de ugyanúgy működik így

    npImg1 = np.asarray(mask1)  #melyikről vegye a mintát
    npImg2 = np.asarray(mask2)
    kd_a = np.argwhere(npImg1 == 255)
    kd_b = np.argwhere(npImg2 == 255)
    kd_a = str(kd_a)
    kd_b = str(kd_b)
    for i in range(15):
        try:
            if kd_a[i] == ' ' and kd1 and not x_a:
                x_a = kd1
                x_a = int(x_a)
                kd1 = ""
            if kd_a[i] == ' ' and kd1 and x_a:
                y_a = kd1
                y_a = int(y_a)
                kd1 = ""
            if kd_a[i] == '0' or kd_a[i] == '1' or kd_a[i] == '2' or kd_a[i] == '3' or kd_a[i] == '4' or kd_a[i] == '5' or kd_a[
                i] == '6' \
                    or kd_a[i] == '7' or kd_a[i] == '8' or kd_a[i] == '9':
                kd1 += kd_a[i]
            cv2.line(vid1, (int(y_a), int(x_a)), (int(y_a), int(x_a)), (0, 0, 255), 10)
            if kd_b[i] == ' ' and kd2 and not x_b:
                x_b = kd2
                x_b = int(x_b)
                kd2 = ""
            if kd_b[i] == ' ' and kd2 and x_b:
                y_b = kd2
                y_b = int(y_b)
                kd2 = ""
            if kd_b[i] == '0' or kd_b[i] == '1' or kd_b[i] == '2' or kd_b[i] == '3' or kd_b[i] == '4' or kd_b[i] == '5' or kd_b[
                i] == '6' \
                    or kd_b[i] == '7' or kd_b[i] == '8' or kd_b[i] == '9':
                kd2 += kd_b[i]
            cv2.line(vid2, (int(y_b), int(x_b)), (int(y_b), int(x_b)), (0, 0, 255), 10)
        except:
            continue
    kd1 = ""
    kd2 = ""
    if x_a != 0 and y_a != 0 and s_1 is False:
        listx_1.append(int(x_a))
        listy_1.append(int(y_a))
        print(listx_1)
    #     if cn_1 > 2:
    #         try:
    #             if listy_1[cn_1] > listy_1[cn_1 - 1]:
    #                 del listy_1[-1]
    #                 del listx_1[-1]
    #                 print(listx_1)
    #                 print(listy_1)
    #
    #                 # listx_1.clear()
    #                 # listy_1.clear()
    #                 # plot()
    #                 # itt most egy-egy egész pályát betesz és kiplottol, de úgy kéne, hogy csak az odautat
    #
    #                 # with open('x.txt', 'a+') as f:
    #                 #     for item in listx:
    #                 #         f.write("%s\n" % item)
    #                 # with open('y.txt', 'a+') as f:
    #                 #     for item in listy:
    #                 #         f.write("%s\n" % item)
    #                 s_1 = True
    #         except:
    #             cn_1 -= 1
    #     cn_1 += 1
    # if x_a == 0 and y_a == 0 and s_1:
    #     s_1 = False

    if x_b != 0 and y_b != 0 and s_2 is False:
        listx_2.append(int(x_b))
        listy_2.append(int(y_b))
    #
    #     if cn_2 > 2:
    #         try:
    #             if listy_2[cn_2] > listy_2[cn_2 - 1]:
    #                 del listy_2[-1]
    #                 del listx_2[-1]
    #                 print(listy_2)
    #                 print(listx_2)
    #
    #                 # with open('x.txt', 'a+') as f:
    #                 #     for item in listx:
    #                 #         f.write("%s\n" % item)
    #                 # with open('y.txt', 'a+') as f:
    #                 #     for item in listy:
    #                 #         f.write("%s\n" % item)
    #                 s_2 = True
    #         except:
    #             cn_2 -= 1
    #     cn_2 += 1
    # if x_b == 0 and y_b == 0 and s_2:
    #     s_2 = False

    # lent(smfr1)
cv2.destroyAllWindows()

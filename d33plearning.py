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

cap1 = cv2.VideoCapture("12_f.mp4")
cap2 = cv2.VideoCapture("12_l.mp4")

# cap1.set(cv2.CAP_PROP_FPS, 60) #60 fpsre állítás
# cap2.set(cv2.CAP_PROP_FPS, 60)

#  az egész kódban a sima x és y változók fel vannak cserélve, kivéve a kiíratós rész
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


def kiir():
    cim1 = str(e) + "y_felso" + txt
    with open(cim1, 'w+') as f:
        for item in listx_felso:
            f.write("%s\n" % item)
    cim1 = str(e) + "x_felso" + txt
    with open(cim1, 'w+') as f:
        for item in listy_felso:
            f.write("%s\n" % item)
    cim1 = str(e) + "y_also" + txt
    with open(cim1, 'w+') as f:
        for item in listx_also:
            f.write("%s\n" % item)
    cim1 = str(e) + "x_also" + txt
    with open(cim1, 'w+') as f:
        for item in listy_also:
            f.write("%s\n" % item)


def plot():
    # AZ EGÉSZET PLOTTOLJA KI MOST!!!!!!!!!!!!!!!!!!!!!
    # plt.plot(list_kiiry_f, list_kiirx_f, 'ro')
    # plt.axis([0, 1280, 720, 0])
    # plt.plot(list_kiiry_l, list_kiirx_l, 'go')
    # plt.axis([0, 1280, 720, 0])
    plt.plot(listy_felso, listx_felso, 'ro')
    plt.axis([0, 1280, 720, 0])
    plt.plot(listy_also, listx_also, 'go')
    plt.axis([0, 1280, 720, 0])
    plt.show()


while 1:  # folyamatosan veszi a jelet
    _, vid1 = cap1.read()  # minden kockát olvas
    _, vid2 = cap2.read()  # minden kockát olvas
    try:
        hsv1 = cv2.cvtColor(vid1, cv2.COLOR_BGR2HSV)  # ez itt a konverzió
        hsv2 = cv2.cvtColor(vid2, cv2.COLOR_BGR2HSV)  # ez itt a konverzió
    except:
        with open("y_11_felső_full.txt", 'w+') as f:
            for item in list_kiirx_f:
                f.write("%s\n" % item)
        with open("x_11_felső_full.txt", 'w+') as f:
            for item in list_kiiry_f:
                f.write("%s\n" % item)
        with open("y_11_alsó_full.txt", 'w+') as f:
            for item in list_kiirx_l:
                f.write("%s\n" % item)
        with open("x_11_alsó_full.txt", 'w+') as f:
            for item in list_kiiry_l:
                f.write("%s\n" % item)
        plot()
    lower_orange1 = np.array([30, 130, 130])
    upper_orange1 = np.array([85, 255, 255])  # új videóknál ez az atom!!!
    lower_orange2 = np.array([30, 130, 130])
    upper_orange2 = np.array([85, 255, 255])

    # lower_orange2 = np.array([30, 130, 130]) #zöld jó
    # upper_orange2 = np.array([85, 255, 255])

    # lower_orange2 = np.array([120, 100, 100]) # lila
    # upper_orange2 = np.array([165, 150, 255])

    mask1 = cv2.inRange(hsv1, lower_orange1, upper_orange1)  # ez maszkolja ki
    mask2 = cv2.inRange(hsv2, lower_orange2, upper_orange2)  # ez maszkolja ki
    # res1 = cv2.bitwise_and(vid1, vid1, mask=mask1)  # a kimaszkolt és igazi kép összeÉSelve.

    external_poly_1_a = np.array([[[0, 0], [150, 0], [110, 720], [0, 720]]], dtype=np.int32)
    external_poly_2_a = np.array([[[0, 0], [1280, 0], [1280, 50], [0, 15]]], dtype=np.int32)
    external_poly_3_a = np.array([[[1040, 0], [1280, 0], [1280, 720], [1030, 720]]], dtype=np.int32)
    external_poly_4_a = np.array([[[0, 660], [1280, 680], [1280, 720], [0, 720]]], dtype=np.int32)

    external_poly_1_b = np.array([[[0, 0], [150, 80], [150, 720], [0, 720]]], dtype=np.int32)
    external_poly_2_b = np.array([[[0, 0], [1280, 0], [1280, 250], [0, 250]]], dtype=np.int32)
    external_poly_3_b = np.array([[[1100, 0], [1280, 0], [1280, 720], [1100, 720]]], dtype=np.int32)
    external_poly_4_b = np.array([[[0, 650], [1280, 650], [1280, 720], [0, 720]]], dtype=np.int32)
    # [[[10, 10], [100, 10], [100, 100], [10, 100]]] kis négyzet
    # https: // stackoverflow.com / questions / 56813343 / masking - out - a - specific - region - in -opencv - python

    cv2.fillPoly(mask1, external_poly_1_a, (0, 0, 0))
    cv2.fillPoly(mask1, external_poly_2_a, (0, 0, 0))
    cv2.fillPoly(mask1, external_poly_3_a, (0, 0, 0))
    cv2.fillPoly(mask1, external_poly_4_a, (0, 0, 0))

    # cv2.fillPoly(mask2, external_poly_1_b, (0, 0, 0))
    # cv2.fillPoly(mask2, external_poly_2_b, (0, 0, 0))
    # cv2.fillPoly(mask2, external_poly_3_b, (0, 0, 0))
    # cv2.fillPoly(mask2, external_poly_4_b, (0, 0, 0))

    # ez a rész csak a videós megjelenítéshez kell
    cv2.fillPoly(vid1, external_poly_1_a, (0, 0, 0))
    cv2.fillPoly(vid1, external_poly_2_a, (0, 0, 0))
    cv2.fillPoly(vid1, external_poly_3_a, (0, 0, 0))
    cv2.fillPoly(vid1, external_poly_4_a, (0, 0, 0))

    # cv2.fillPoly(vid2, external_poly_1_b, (0, 0, 0))
    # cv2.fillPoly(vid2, external_poly_2_b, (0, 0, 0))
    # cv2.fillPoly(vid2, external_poly_3_b, (0, 0, 0))
    # cv2.fillPoly(vid2, external_poly_4_b, (0, 0, 0))

    cv2.line(vid1, (int(y_a), int(x_a)), (int(y_a), int(x_a)), (0, 0, 255), 15)
    cv2.line(vid2, (int(y_b), int(x_b)), (int(y_b), int(x_b)), (255, 0, 0), 15)

    x_a = 0
    y_a = 0
    x_b = 0
    y_b = 0
    cv2.imshow('vid1', vid1)  # mit mutat
    cv2.imshow('vid2', vid2)
    cv2.waitKey(1)  # k = cv2.waitKey(1) & 0xFF #itt ez nem tudom mi volt, de ugyanúgy működik így

    npImg1 = np.asarray(mask1)  # melyikről vegye a mintát
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
            if kd_a[i] == '0' or kd_a[i] == '1' or kd_a[i] == '2' or kd_a[i] == '3' or kd_a[i] == '4' or kd_a[i] == '5' \
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
            if kd_b[i] == '0' or kd_b[i] == '1' or kd_b[i] == '2' or kd_b[i] == '3' or kd_b[i] == '4' or kd_b[i] == '5' \
                    or kd_b[i] == '6' or kd_b[i] == '7' or kd_b[i] == '8' or kd_b[i] == '9':
                kd2 += kd_b[i]
        except:
            pass
    kd1 = ""
    kd2 = ""
    if x_a != 0 and y_a != 0 and s_1 is False:
        listx_felso.append(int(x_a))
        listy_felso.append(int(y_a))
        listx_also.append(int(x_b))
        listy_also.append(int(y_b))

        if cn_1 > 1:
            try:
                if listy_felso[cn_1] > listy_felso[cn_1 - 1]:
                    del listy_felso[-1]
                    del listy_also[-1]
                    del listx_felso[-1]
                    del listx_also[-1]
                    kiir()
                    e += 1
                    s_1 = True
                    s_2 = True
                    cn_1 = 0
            except:
                cn_1 -= 1
        cn_1 += 1

    if x_a == 0 and y_a == 0 and s_1:
        print("y felső: ", listx_felso)  # éppen az aktuális ütés
        print("x felső: ", listy_felso)
        print("y alsó: ", listx_also)  # éppen az aktuális ütés
        print("x alsó: ", listy_also)
        list_kiirx_f += listx_felso  # az összes ütés
        list_kiiry_f += listy_felso
        list_kiirx_l += listx_also  # az összes ütés
        list_kiiry_l += listy_also
        plot()
        listx_felso = []  # kinullázza az aktuális listát
        listy_felso = []
        listx_also = []  # kinullázza az aktuális listát
        listy_also = []

        s_1 = False
    #
    if x_b == 0 and y_b == 0 and s_2:
        s_2 = False

cv2.destroyAllWindows()

import cv2
import matplotlib.pyplot as plt
import numpy as np


def preprocessing(img):
    #img_bgr = cv2.merge((img, img, img))

    img_bgr = cv2.imread(img)

    filter = np.array([[0, -1, 0], [-1, 6, -1], [0, -1, 0]])
    edge = cv2.filter2D(src=img_bgr, kernel=filter, ddepth=-1)

    # Convert from BGR to YUV
    img_yuv = cv2.cvtColor(edge, cv2.COLOR_BGR2YUV)

    # Converting directly back from YUV to BGR results in an (almost) identical image
    B, G, R = cv2.split(img_yuv)

    output1_R = cv2.equalizeHist(R)
    output1_G = cv2.equalizeHist(G)
    output1_B = cv2.equalizeHist(B)

    equ = cv2.merge((output1_B, output1_G, output1_R))
    median = cv2.medianBlur(equ, 1)

    plt.imshow(img_bgr)
    plt.show()

    plt.imshow(edge)
    plt.show()

    plt.imshow(img_yuv)
    plt.show()

    plt.imshow(equ)
    plt.show()

    plt.imshow(median)
    plt.show()

    horizontalStack = np.concatenate((img_bgr, edge, img_yuv, equ, median), axis=1)
    plt.imshow(horizontalStack,cmap="gray")

    plt.axis("off")
    plt.show()



def yuv_bgr(img):
    img = cv2.imread(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    greenMask = cv2.inRange(hsv, (26, 10, 30), (97, 100, 255))

    hsv[:, :, 1] = greenMask

    back = cv2.cvtColor(hsv, cv2.COLOR_YUV2BGR)

    plt.imshow(back)
    plt.show()

def show(img):

    plt.imshow(img,cmap="gray")
    plt.show()

#yuv_bgr("0e3860fcb09afc17ac43bfcef8f7663b_crop.jpg")

#show("7ec445ed6034ed610a43877369354aa4.jpg")
preprocessing("0e3860fcb09afc17ac43bfcef8f7663b_crop.jpg")
#preprocess("1e70699bd3ec25487be502c96a4ffbaf.jpg")
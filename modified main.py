import cv2 as cv
import numpy as np

def color_mask(image, up,down):
    Pnum = cv.countNonZero( cv.inRange(image, down, up))
    return Pnum
def deleteText(c, imageA):
    cimg = np.zeros_like(imageA)
    cv.drawContours(cimg, [c], -1, (255, 255, 255), thickness=2)
    newImage = cimg.copy()
    h, w = cimg.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv.floodFill(newImage, mask, (0, 0), 255);
    newImage_inv = cv.bitwise_not(newImage)
    out = cimg | newImage_inv
    for i in range(0, imageA.shape[0]):
        for j in range(0, imageA.shape[1]):
            if (out[i, j][0] != 255 or out[i, j][1] != 255 or out[i, j][2] != 255):
                imageA[i, j] = 0

    h, w = imageA.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv.floodFill(imageA, mask, (0, 0), (255, 255, 255));

    grayB = cv.cvtColor(imageA, cv.COLOR_BGR2GRAY)
    return grayB,imageA
def rescale(W,image):

    height, width, depth = image.shape
    imgzcale = W / width
    newX, newY = image.shape[1] * imgzcale, image.shape[0] * imgzcale
    newimage = cv.resize(image, (int(newX), int(newY)))

    return newimage
def findc(grayA):
    blur = cv.medianBlur(grayA, 7)
    laplacian = cv.Laplacian(blur, cv.CV_8UC1)
    contours, hierarchy = cv.findContours(laplacian, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    c = max(contours, key=cv.contourArea)


    epsilon = 0.025 * cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, epsilon, True)

    ret, thresholded = cv.threshold(blur, 127, 255, cv.THRESH_BINARY)
    invetred_image = cv.bitwise_not(thresholded)
    contours2, hierarchy2 = cv.findContours(invetred_image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    c2=min(contours2, key=cv.contourArea)
    approx2 = cv.approxPolyDP(c2, epsilon, True)

    cv.imshow("d",invetred_image)
    cv.waitKey(0)

    return approx, c,contours2,c2,approx2


if __name__ == '__main__':

    input_image = cv.imread("circles/no.jpg")
    input_image = rescale(300, input_image)
    image_size = input_image .shape[0] * input_image.shape[1]
    image_gray = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)
    hsv=cv.cvtColor(input_image, cv.COLOR_BGR2HSV)

    yellow_number = color_mask(hsv, np.array([35, 255, 255]), np.array([15, 100, 100]))
    green_number = color_mask(hsv, np.array([90, 255, 255]), np.array([30, 100, 50]))
    brown_number = color_mask(hsv, np.array([20, 255, 200]), np.array([10, 100, 20]))
    red_number1 = color_mask(hsv, np.array([10, 255, 255]), np.array([0, 50, 50]))
    red_number2 = color_mask(hsv, np.array([180, 255, 255]), np.array([170, 50, 50]))
    red_number= red_number1 + red_number2
    blue_number = color_mask(hsv, np.array([130, 255, 255]), np.array([80, 50, 50]))

    approximated, c,xxx,xxxxx,xxxx = findc(image_gray)
    grayB, input_image = deleteText(c, input_image)
    xx, c, contours2 ,xxxxxxx,xxxxxxxxx= findc(grayB)
    hsv2=cv.cvtColor(input_image, cv.COLOR_BGR2HSV)
    black_number = color_mask(hsv2, np.array([180, 255, 40]), np.array([0, 0, 0]))

    if len(approximated) == 4 and yellow_number > 100:
        print("exit")
    elif green_number>100 and len(approximated) == 4:
        print("major")
    elif brown_number / image_size > 0.2:
        print("tour")
    elif len(approximated) == 3 and yellow_number>100 and  green_number>100 :
        print("traffic light")
    elif len(approximated) > 4 and blue_number > 100 and red_number > 100:
        print("no parking")
    else:
        if len(approximated) == 4 and blue_number>100:
            print("free road")
        else:
            if blue_number/image_size < 0.07 and  red_number/image_size < 0.07:
                if len(approximated) > 7:
                    print("speed_limit")
                else:
                    print("local_destination")
            elif len(approximated) == 3:
                if len(contours2) == 3:
                    print("bump")
                elif black_number> 200 :
                    print("dangerous_descendent ")
                else:
                    print("give way")
            else:
                xx, xxxxx, contours, c2, approximated2 = findc(grayB)

                center = cv.moments(c2)
                center_x = int(center["m10"] / (center["m00"]+1))
                center_y = int(center["m01"] / (center["m00"]+1))
                count_right = count_left = count_top = 0

                for i in range(len(approximated2)):
                   if approximated2[i][0][0] > center_x:
                       count_right = count_right + 1
                   elif approximated2[i][0][0] < center_x:
                       count_left = count_left + 1
                   if  approximated2[i][0][1] < center_y:
                       count_top = count_top + 1

                if blue_number/image_size>0.3:
                        if count_top > count_left and count_top > count_right:
                            print("go straight")
                        elif count_right > count_left:
                            print("go right")
                        elif count_left > count_right:
                            print("go left")

                else:
                    if len(approximated) > 4:
                        print("stop")
                    else:
                        print("no")
import cv2 as cv
import numpy as np

if __name__ == '__main__':
    imageA = cv.imread("")
    #rescaling the image
    W = 300.
    height, width, depth = imageA.shape
    imgScale = W / width
    newX, newY = imageA.shape[1] * imgScale, imageA.shape[0] * imgScale
    imageA = cv.resize(imageA, (int(newX), int(newY)))
    #convert image to gray and hsv color spaces
    grayA = cv.cvtColor(imageA, cv.COLOR_BGR2GRAY)
    hsv=cv.cvtColor(imageA, cv.COLOR_BGR2HSV)

    ##yellow detiction
    lower_yellow=np.array([15, 100, 100])
    upper_yellow=np.array([35, 255, 255])
    mask=cv.inRange(hsv,lower_yellow,upper_yellow)
    px=mask.shape[0]*mask.shape[1]
    whitey= cv.countNonZero(mask)

    ##green detiction
    lower_green=np.array([30, 100, 50])
    upper_green=np.array([90, 255, 255])
    mask2=cv.inRange(hsv,lower_green,upper_green)
    px2=mask.shape[0]*mask.shape[1]
    whiteg= cv.countNonZero(mask2)

    ##brown detiction
    lower_brown=np.array([10, 100, 20])
    upper_brown=np.array([20, 255, 200])
    mask3=cv.inRange(hsv,lower_brown,upper_brown)
    px3=mask.shape[0]*mask.shape[1]
    whiteb= cv.countNonZero(mask3)

    ##red detiction
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask4= cv.inRange(hsv, lower_red, upper_red)
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask5 = cv.inRange(hsv, lower_red, upper_red)
    mask6 = mask4+mask5
    whiter= cv.countNonZero(mask6)

    ##blue detiction
    lower_blue=np.array([80,50,50])
    upper_blue=np.array([130,255,255])
    mask7=cv.inRange(hsv,lower_blue,upper_blue)
    px7=mask.shape[0]*mask.shape[1]
    whiteblue= cv.countNonZero(mask7)


    #detected  by  color
    if whitey/px > 0.25 :
        print("exit")
    elif whiteg/px2 >0.2 :
        print("major")
    elif whiteb / px3 > 0.2:
        print("tour")
    elif  whitey>100 and  whiteg>100 :
        print("traffic light")
    elif whiteblue/px > 0.2 and whiter/px > 0.2:
        print("park")
    elif  whiteblue/px <0.07 and  whiter/px<0.07 :
        #to detect the images whis not containing blue and rde
        thresholded = cv.adaptiveThreshold(grayA, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
        kernel = np.ones((6, 6), np.uint8)
        erosion = cv.dilate(thresholded, kernel, iterations=1)

        # cv.imshow("d",erosion)
        # cv.waitKey(0)

        edged = cv.Canny(erosion, 255, 255, 255)
        contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL,  cv.CHAIN_APPROX_TC89_L1)
        cv.drawContours(imageA, contours, -1, (0,255,0), 3)

        if ( len(contours)>11 ):
            print("endspeed limit")
        else:
            print("local destination")

    else:
        #find contours
        ret, thresh1 = cv.threshold(grayA, 127, 255, cv.THRESH_BINARY)
        blur=cv.medianBlur(grayA,7)
        laplacian = cv.Laplacian(blur, cv.CV_8UC1)
        contours, hierarchy = cv.findContours(laplacian, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        c = max(contours, key=cv.contourArea)
        epsilon = 0.02 * cv.arcLength(c , True)
        approx = cv.approxPolyDP(c , epsilon, True)

        if(len(approx)==4):
            print("free road")
        else:
            ##to delete text and always put background to white
            cimg = np.zeros_like(imageA)
            cv.drawContours(cimg, [c], -1,(255,255,255), thickness=2)
            im_floodfill = cimg.copy()
            h, w = cimg.shape[:2]
            mask = np.zeros((h + 2, w + 2), np.uint8)
            cv.floodFill(im_floodfill, mask, (0, 0), 255);
            im_floodfill_inv = cv.bitwise_not(im_floodfill)

            im_out = cimg | im_floodfill_inv
            for i in range(0, imageA.shape[0]):
                for j in range(0, imageA.shape[1]):
                    if(im_out[i, j][0]!=255 or im_out[i, j][1]!=255  or im_out[i, j][2]!=255 ):
                        imageA[i, j] = 0
            # cv.imshow("d", imageA)
            # cv.waitKey(0)

            h, w = imageA.shape[:2]
            mask = np.zeros((h + 2, w + 2), np.uint8)
            # Floodfill from point (0, 0)
            cv.floodFill(imageA, mask, (0, 0), (255,255,255));
            grayB = cv.cvtColor(imageA, cv.COLOR_BGR2GRAY)

            # cv.imshow("ddddddddddddd", imageA)
            # cv.waitKey(0)
            # print(len(approx))

            #to detect traiangles
            if(len(approx)==3):
                ret, thresh1 = cv.threshold(grayB, 127, 255, cv.THRESH_BINARY)
                inv = cv.bitwise_not(thresh1)
                blur = cv.medianBlur(grayB, 7)
                laplacian = cv.Laplacian(blur, cv.CV_8UC1)
                contours, hierarchy = cv.findContours(inv, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

                ##black detiction
                lower_yellow = np.array([0, 0, 0])
                upper_yellow = np.array([180, 255, 40])
                hsv = cv.cvtColor(imageA, cv.COLOR_BGR2HSV)
                mask = cv.inRange(hsv, lower_yellow, upper_yellow)
                whiter = cv.countNonZero(mask)
                px7 = mask.shape[0] * mask.shape[1]

                # cv.imshow("d", mask)
                # cv.waitKey(0)

                if(len(contours)==3):
                    print("bump")
                elif(whiter/px7>0.03):
                    print("BumbyRoad")
                else:
                    print("give")
            else:
                ret, thresh1 = cv.threshold(grayB, 127, 255, cv.THRESH_BINARY)
                blur = cv.medianBlur(thresh1, 11)
                inv = cv.bitwise_not(blur)
                contours, hierarchy = cv.findContours(inv, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
                c = min(contours, key=cv.contourArea)
                approx = cv.approxPolyDP(c, epsilon, True)

                # cv.drawContours(imageA,approx,-1,(0,255,0),4)
                # cv.imshow("d", imageA)
                # cv.waitKey(0)

                M = cv.moments(c)
                cX = int(M["m10"] / (M["m00"]+1))
                cY = int(M["m01"] /( M["m00"]+1))

                # cv.circle(imageA, (cX, cY), 3, (0, 0, 0), -1)
                # cv.imshow("d", imageA)
                # cv.waitKey(0)

                #to count nuber of points on right and left and top
                count=0
                count2=0
                count3=0
                print(cY)
                for i in range(len(approx)):
                   if approx[i][0][0]> cX:
                       count=count+1
                   if approx[i][0][0] < cX:
                       count2=count2+1
                   if  approx[i][0][1] < cY:
                       count3=count3+1

                #to detect blue curciles
                if(whiteblue/px7>0.3):
                    if(len(contours)>4):
                        # print(len(contours))
                        print("cannot be recognized")
                    else:
                        # print(count)
                        # print(count3)
                        if count3>count2 and count3>count:
                            print("up")
                        elif count > count2:
                            print("right")
                        elif count2 >count:
                            print("left")
                # to detect red ones
                else:
                    # print(len(approx))
                    if len(approx)==4:
                        print("no")
                    else:
                        print("stop")
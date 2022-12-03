import numpy as np
import cv2
import os
import sys
import argparse
import violaJones
import hough

def hasMostlyRed(image, threshold):
    found = False
    newImage = np.array(image)
    img_hsv = cv2.cvtColor(newImage, cv2.COLOR_BGR2HSV)

    # lower mask (0-10)
    lower_red = np.array([0,50,50])
    upper_red = np.array([25,255,255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([160,50,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # join my masks
    mask = mask0 | mask1

    # set my output img to zero everywhere except my mask
    newImage[np.where(mask==0)] = 255
    cv2.imshow('red', newImage)
    cv2.waitKey()
    
    return newImage

    # newImage = newImage[...,2]    
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    # closing = cv2.morphologyEx(newImage, cv2.MORPH_CLOSE, kernel)

    # newImage[np.where(mask!=0)] = 1
    # pixels = np.sum(newImage)

    # val = pixels / (newImage.shape[0] * newImage.shape[1])

    # return val >= threshold

    # newImage[np.where(newImage==1)] = 255
    # print("pixels", pixels, newImage.shape[0] * newImage.shape[1], pixels / (newImage.shape[0] * newImage.shape[1]))    
    # cv2.imshow('red', newImage)
    # cv2.waitKey()

def hasRectangle(image):
    found = False
    newImage = np.array(image)
    
    # obtain binary image        
    blur = cv2.GaussianBlur(newImage, (5, 5), 2, 2)
    gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
    adaptive = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,57,11)

    WHITE_MIN = np.array([0, 200, 0],np.uint8)
    WHITE_MAX = np.array([180, 255, 255],np.uint8)
    hls = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)        
    white = cv2.inRange(hls, WHITE_MIN, WHITE_MAX)

    canny = cv2.Canny(gray, 100, 200)

    thresh = adaptive & (canny | white)

    # fill rectangular contours
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)        
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, (255,255,255), -1)

    # perform morph open
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # draw rectangles
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)

        if w > h*2 and w > newImage.shape[1]/2 and w < newImage.shape[1]: 
            found = True
            cv2.rectangle(newImage, (x, y), (x + w, y + h), (255,0,0), 3)

    # cv2.imshow('thresh', thresh)
    # cv2.imshow('opening', opening)
    # cv2.imshow('image', mini)
    # cv2.waitKey()

    return found, newImage


def removingDuplicateCircles(houghspace, delta, threshold):
    newHough = {}

    for y in range(houghspace.shape[0]):
        for x in range(houghspace.shape[1]):
            
            if sum(i > threshold for i in houghspace[y][x]) == 0:
                continue

            can = True
            for (ty, tx) in newHough.keys():
                if x-delta <= tx and tx <= x+delta and y-delta <= ty and ty <= y+delta:
                    can = False
                    break

            if can:
                accum, count = 0.0, 0.0
                for r in range(houghspace.shape[2]):
                    if houghspace[y][x][r] > threshold:
                        accum += r
                        count += 1.0

                if count:                    
                    newHough[(y, x)] = round(accum / count) 
            
                    
    return newHough

def getBoxFromCircle(x, y, r, delta):
    new_y = max(y - r - delta, 0)
    new_x = max(x - r - delta, 0)

    width = heigth = r*2 + delta*2
    return (new_x, new_y, width, height)

def correspondenceBoxCircle(bx, by, width, height, cx, cy, circle_radius):
    box_radius = (width + height) / 2.0

    centreSimilarity = math.e ** (-np.linalg.norm(np.subtract(np.array([cx, cy]), np.array([bx, by]))))    
    radiusSimilarity = math.e ** (-np.linalg.norm(np.subtract(np.array([circle_radius]), np.array([box_radius]))))
    similarity = (centreSimilarity + radiusSimilarity) / 2.0

    return similarity > 0.5

def splitting(boxes, circles):
    common = set()
    removeBoxes, removeCircles = set(), set()
    for (boxx, boxy, boxWidth, boxHeight) in boxes:
        for (circley, circlex) in circles.keys():

            if (circley, circlex) not in removeCircles:
                circleRad =  circles[(circley, circlex)]
                
                if correspondenceBoxCircle(boxx, boxy, boxWidth, boxHeight, circlex, circley, circleRad):
                    new_x = round((boxx + circlex) / 2.0)
                    new_y = round((boxy + circley) / 2.0)
                    newWidth = max(boxWidth, circleRad) # increase rad?? pre
                    newHeight = max(boxHeight, circleRad)
                    common.add(new_x, new_y, newWidth, newHeight)

                    removeBoxes.add((boxx, boxy, boxWidth, boxHeight))
                    removeCircles.add((circley, circlex))

    for i in removeBoxes:
        boxes.remove(i)

    for i in removeCircles:
        del circles[i]

    return boxes, common, circles



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='no entry sign detection')
    parser.add_argument('-name', '-n', type=str, default='No_entry/NoEntry0.bmp')
    args = parser.parse_args()

    imageName = args.name
    idx = imageName[16:18]
    print(idx)
    cascade_name = "cascade.xml"

    # ignore if no such file is present.
    if (not os.path.isfile(imageName)) or (not os.path.isfile(cascade_name)):
        print('No such file')
        sys.exit(1)

    # read Input Image
    frame = cv2.imread(imageName, 1)

    processing = hasMostlyRed(frame, 69)



    # Z = frame.reshape((-1,3))
    
    # # convert to np.float32
    # Z = np.float32(Z)
    
    # # define criteria, number of clusters(K) and apply kmeans()
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # K = 7
    # ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
    # # Now convert back into uint8, and make original image
    # center = np.uint8(center)
    # res = center[label.flatten()]
    # res2 = res.reshape((frame.shape))

    # cv2.imshow('res2',res2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # processing = res2


    # ignore if image is not array.
    if not (type(frame) is np.ndarray):
        print('Not image data')
        sys.exit(1)

    # applying violaJones

    model = violaJones.loadClassifier(cascade_name) # blur?
    #groundTruth_set = readGroundtruth('groundtruth.txt')
    predictions_set = violaJones.detect(processing, model)
    #frame = display(frame, groundTruth_set, (0, 0, 255))
    boxes = violaJones.display(frame, predictions_set, (0, 255, 0))
    #assess(groundTruth_set, predictions_set)
    cv2.imwrite( "boxes.jpg", boxes )

    # grayscale
    if processing.shape[2] >= 3:
        processing = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )


    # # applying hough
    # output = hough.houghCircle(processing, 40, 10, 120)
    # output = removingDuplicateCircles(output, 200, 10)
    # circles = hough.displayHoughCircles(frame, output)
    # cv2.imwrite( "circle" + idx + ".png", circles )

    # delta = 0
    # lines = np.array(frame)
    # for (x, y, width, height) in predictions_set:
    #     start_point = [x - delta, y - delta]
    #     end_point = [x + width + delta, y + height + delta]

    #     # make sure to not go over the bounds of the image
    #     start_point[0] = max(start_point[0], 0)
    #     start_point[1] = max(start_point[1], 0)
    #     end_point[0] = min(end_point[0], frame.shape[1]-1)
    #     end_point[1] = min(end_point[1], frame.shape[0]-1)       

    #     # creating the mini image that will contain a small section of the original image
    #     rows, cols = end_point[1] - start_point[1] + 1, end_point[0] - start_point[0] + 1
    #     mini = np.zeros((rows, cols, 3), np.uint8)
    #     for r in range(rows):
    #         for c in range(cols):
    #             mini[r, c] = frame[r + start_point[1], c + start_point[0]]

    #     #found, mini = hasRectangle(mini)
    #     cond = hasMostlyRed(mini, 0.267)

    #     # pasting the mini section back onto the image
    #     for r in range(rows):
    #         for c in range(cols):
    #             if cond:
    #                 lines[r + start_point[1], c + start_point[0]] = mini[r, c]
    #             else:
    #                 lines[r + start_point[1], c + start_point[0]] = 0

        
    # cv2.imwrite( "lines.png", lines )

    # for each hough circle, find the closest bounding box whose box centre is closest to its own circle centre
    # once we found the bounding box, make sure that the circle centre is really within that bounding box
    # resize the bounding box depending on the size of the circle
    # at the end, all bounding boxes who have been found at step 2. will be drawn 




# must not have rectangles in each circle
# better have less false positives than more true positives


# finding more true positives + reducing false negatives - hough circle, viola jones
# less false positives - the many checks


# if viola is wrong but really close???
# viola box in box
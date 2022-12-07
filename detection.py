import numpy as np
import cv2
import os
import sys
import argparse
import violaJones
import hough

def hasMostlyRed(image, threshold):
    newImage = np.array(image)
    img_hsv = cv2.cvtColor(newImage, cv2.COLOR_BGR2HSV)

    # lower mask (0-10)
    lower_red = np.array([0,50,50])
    upper_red = np.array([3,255,255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([175,50,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # join my masks
    mask = mask0 | mask1

    # set my output img to processero everywhere except my mask
    newImage[np.where(mask==0)] = 0
    newImage[np.where(mask!=0)] = 1

    # obtain grayscale
    newImage = newImage[...,2]    
    
    # close any gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    closing = cv2.morphologyEx(newImage, cv2.MORPH_CLOSE, kernel)
    
    pixels = np.sum(newImage)
    val = pixels / (newImage.shape[0] * newImage.shape[1])
    return val >= threshold

def hasRectangle(image, threshold1, threshold2):
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

    if len(cnts) > 1:
        return False, newImage

    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        ratio = w*h / (newImage.shape[0]*newImage.shape[1])
        if w > h*2 and threshold1 < ratio and ratio < threshold2: 
            found = True
            cv2.rectangle(newImage, (x, y), (x + w, y + h), (255,0,0), 3)

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

def removeUselessBoxes(predictions_set):
    predictions_list = list(predictions_set)
    predictions_list.sort(key=lambda x: x[2] * x[3], reverse=True)
    output = set()

    for (x, y, width, height) in predictions_list:
        nope = False
        lasto = (x + width, y + height)

        for (x1, y1, width1, height1) in output:
            lasto1 = (x1 + width1, y1 + height1)

            if x1 <= x and lasto[0] <= lasto1[0] and y1 <= y and lasto[1] <= lasto1[1]:
                nope = True
                break

        if not nope:
            output.add((x, y, width, height))

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='no entry sign detection')
    parser.add_argument('-name', '-n', type=str, default='No_entry/NoEntry0.bmp')
    args = parser.parse_args()

    imageName = args.name
    cascade_name = "cascade.xml"

    # ignore if no such file is present.
    if (not os.path.isfile(imageName)) or (not os.path.isfile(cascade_name)):
        print('No such file')
        sys.exit(1)

    # read Input Image
    frame = cv2.imread(imageName, 1)

    imageName = imageName.split("/")[-1]

    # ignore if image is not array.
    if not (type(frame) is np.ndarray):
        print('Not image data')
        sys.exit(1)

    # [*] 1. viola jones + cascade classifier
    pre_vj = np.array(frame)
    img_hsv = cv2.cvtColor(pre_vj, cv2.COLOR_BGR2HSV)

    # lower mask (0-10)
    lower_red = np.array([0,50,50])
    upper_red = np.array([25,255,255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
    
    # upper mask (170-180)
    lower_red = np.array([160,50,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # join masks
    mask = mask0 | mask1

    # set my output img to white everywhere except my mask
    pre_vj[np.where(mask==0)] = 255

    model = violaJones.loadClassifier(cascade_name)
    predictions_set = violaJones.detect(pre_vj, model)
    predictions_set = removeUselessBoxes(predictions_set)

    
    delta = 10
    to_remove = set()
    for (x, y, width, height) in predictions_set:
        start_point = [x - delta, y - delta]
        end_point = [x + width + delta, y + height + delta]

        # make sure to not go over the bounds of the image
        start_point[0] = max(start_point[0], 0)
        start_point[1] = max(start_point[1], 0)
        end_point[0] = min(end_point[0], frame.shape[1]-1)
        end_point[1] = min(end_point[1], frame.shape[0]-1)       

        # creating the mini image that will contain a small section of the original image
        rows, cols = end_point[1] - start_point[1] + 1, end_point[0] - start_point[0] + 1
        mini = np.processeros((rows, cols, 3), np.uint8)
        for r in range(rows):
            for c in range(cols):
                mini[r, c] = frame[r + start_point[1], c + start_point[0]]

        # K-means (removed)

        # process = mini.reshape((-1,3))
        # process = np.float32(process)
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        # K = 4
        # ret,label,center=cv2.kmeans(process,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        # center = np.uint8(center)
        # res = center[label.flatten()]
        # res2 = res.reshape((mini.shape))

        # grayscale
        gray = cv2.cvtColor( mini, cv2.COLOR_BGR2GRAY )

        # [*] 2. Applying Hough Circle
        circles = hough.houghCircle(gray, 35, 17, 120)
        cond3, circlesImg = hough.displayHoughCircles(mini, circles, 10) 
 
        # [*] 3. Mostly red check - Colour Masking, Closing Morphology
        cond1 = hasMostlyRed(mini, 0.1)

        # [*] 4. Has white rectangle check - Adaptive Thresholding, Finding Contours, Opening Morphology
        cond2, mini = hasRectangle(mini, 0.1, 0.25)

        if not(cond3 or (not cond3 and cond1 and cond2)):
            to_remove.add((x, y, width, height))

    for i in to_remove:
        predictions_set.remove(i)

    groundTruth_set = violaJones.readGroundtruth('groundtruth.txt', imageName)
    boxes = violaJones.display(frame, groundTruth_set, (0, 0, 255))
    boxes = violaJones.display(boxes, predictions_set, (0, 255, 0))
    violaJones.assess(groundTruth_set, predictions_set)
    cv2.imwrite( "detected.jpg", boxes )
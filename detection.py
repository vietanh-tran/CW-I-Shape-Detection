import numpy as np
import cv2
import os
import sys
import argparse
import violaJones
import hough

def intersection(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2

    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])

    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return (x0, y0)

# def getLineIntersections(houghSpace):
#     intersections = set()

#     for idx, (m1, c1) in enumerate(lines):
#         for (m2, c2) in lines[idx+1:]:
#             i = getLineIntersection((m1, c1), (m2, c2))
#             if i != None:
#                 intersections.add(i)

#     return intersections


# def checkLineIntersections(x, y, width, height, intersections, mini, maxi):
#     start_point = (x, y)
#     end_point = (x + width, y + height)

#     count = 0
#     for (ix, iy) in intersections:
#         if start_point[0] <= ix and ix <= end_point[0] and 
#             start_point[1] <= iy and iy <= end_point[1]:
#             count += 1

#     return mini <= count and count <= maxi


# def checkColours(image, x, y, width, height):
#     start_point = (x, y)
#     end_point = (x + width, y + height)

#     red, white = 0, 0
#     for row in range(start_point[1], end_point[1] + 1):
#         for col in range(start_point[0], end_point[0] + 1):
#             if is red:
#                 red += 1
#             elif is white:
#                 white += 1

#     return True or False depending on the proportion of red and white



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

    # ignore if image is not array.
    if not (type(frame) is np.ndarray):
        print('Not image data')
        sys.exit(1)

    # # blurring
    # processing = cv2.GaussianBlur(frame, (5, 5), 2, 2)

    # # grayscale
    # if processing.shape[2] >= 3:
    # 	processing = cv2.cvtColor( processing, cv2.COLOR_BGR2GRAY )
    # 	processing = processing.astype(np.float32)
    # else:
    # 	processing = processing.astype(np.float32)

    # applying violaJones

    model = violaJones.loadClassifier(cascade_name) # blur?
    #groundTruth_set = readGroundtruth('groundtruth.txt')
    predictions_set = violaJones.detect(frame, model)
    #frame = display(frame, groundTruth_set, (0, 0, 255))
    boxes = violaJones.display(frame, predictions_set, (0, 255, 0))
    #assess(groundTruth_set, predictions_set)
    cv2.imwrite( "boxes.jpg", boxes )

    # applying hough
    #output = hough.houghCircle(gray_image, 50, 10, 80)
    #circles = hough.displayHoughCircles(frame, output, 10)
    #cv2.imwrite( "circle.png", circles )


    delta = 0
    lines = np.array(frame)
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
        mini = np.zeros((rows, cols, 3), np.uint8)
        for r in range(rows):
            for c in range(cols):
                mini[r, c] = frame[r + start_point[1], c + start_point[0]]

        # detecting rectangles
        # obtain binary image
        
        blur = cv2.GaussianBlur(mini, (5, 5), 2, 2)
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

            if w > h*2 and w > mini.shape[1]/2 and w < mini.shape[1]: 
                cv2.rectangle(mini, (x, y), (x + w, y + h), (255,0,0), 3)

        # cv2.imshow('thresh', thresh)
        # cv2.imshow('opening', opening)
        # cv2.imshow('image', mini)
        # cv2.waitKey()

        # pasting the mini section back onto the image
        for r in range(rows):
            for c in range(cols):
                lines[r + start_point[1], c + start_point[0]] = mini[r, c]

        


    cv2.imwrite( "lines.png", lines )

    # for each hough circle, find the closest bounding box whose box centre is closest to its own circle centre
    # once we found the bounding box, make sure that the circle centre is really within that bounding box
    # resize the bounding box depending on the size of the circle
    # at the end, all bounding boxes who have been found at step 2. will be drawn 




# must not have rectangles in each circle
# better have less false positives than more true positives


# finding more true positives + reducing false negatives - hough circle, viola jones
# less false positives - the many checks

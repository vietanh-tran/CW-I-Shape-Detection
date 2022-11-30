import numpy as np
import cv2
import os
import sys
import argparse
import violaJones
import hough

# def getLineIntersection(line1, line2):
#     m1, c1 = line1
#     m2, c2 = line2



#     return None


# def getLineIntersections(lines):
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

        start_point[0] = max(start_point[0], 0)
        start_point[1] = max(start_point[1], 0)
        end_point[0] = min(end_point[0], frame.shape[1]-1)
        end_point[1] = min(end_point[1], frame.shape[0]-1)       

        rows, cols = end_point[1] - start_point[1] + 1, end_point[0] - start_point[0] + 1
        mini = np.zeros((rows, cols, 3), np.uint8)

        for r in range(rows):
            for c in range(cols):
                mini[r, c] = frame[r + start_point[1], c + start_point[0]]


        # blurring
        processing = cv2.GaussianBlur(mini, (5, 5), 2, 2)

        # grayscale
        if processing.shape[2] >= 3:
            processing = cv2.cvtColor( processing, cv2.COLOR_BGR2GRAY )
            processing = processing.astype(np.float32)
        else:
            processing = processing.astype(np.float32)


        output = hough.houghLine(processing, 60, 0.01)
        mini = hough.displayHoughLines(mini, output, 1)
        for r in range(rows):
            for c in range(cols):
                lines[r + start_point[1], c + start_point[0]] = mini[r, c]

        
    print(len(predictions_set))


    cv2.imwrite( "lines.png", lines )

    # for each hough circle, find the closest bounding box whose box centre is closest to its own circle centre
    # once we found the bounding box, make sure that the circle centre is really within that bounding box
    # resize the bounding box depending on the size of the circle
    # at the end, all bounding boxes who have been found at step 2. will be drawn 

    
import numpy as np
import cv2
import os
import sys
import argparse

# ground_truth = [x, y, width, height]
# prediction = [x, y, width, height]
def iou(ground_truth, prediction):
    gt_startPoint, gt_endPoint = (ground_truth[0], ground_truth[1]), (ground_truth[0] + ground_truth[2], ground_truth[1] + ground_truth[3]) 
    pred_startPoint, pred_endPoint = (prediction[0], prediction[1]), (prediction[0] + prediction[2], prediction[1] + prediction[3]) 
    
    # Intersection coordinates
    intersection_startPoint = (max(gt_startPoint[0], pred_startPoint[0]), max(gt_startPoint[1], pred_startPoint[1]))
    intersection_endPoint = (min(gt_endPoint[0], pred_endPoint[0]), min(gt_endPoint[1], pred_endPoint[1]))

    # Area of Ground Truth Box
    gt_height = gt_endPoint[1] - gt_startPoint[1] + 1
    gt_width = gt_endPoint[0] - gt_startPoint[0] + 1
    area_of_gt = gt_width * gt_height

    # Area of Predicted Box
    pred_height = pred_endPoint[1] - pred_startPoint[1] + 1
    pred_width = pred_endPoint[0] - pred_startPoint[0] + 1
    area_of_pred = pred_width * pred_height
    
    # Area of Union and Area of Intersection
    intersection_height = max(intersection_endPoint[1] - intersection_startPoint[1] + 1, 0) 
    intersection_width = max(intersection_endPoint[0] - intersection_startPoint[0] + 1, 0)
    area_of_intersection = intersection_width * intersection_height
    area_of_union = area_of_gt + area_of_pred - area_of_intersection

    iou = area_of_intersection / area_of_union
    return iou

def detect(frame, model):
    predictions_set = set()

	# 1. Prepare Image by turning it into Grayscale and normalising lighting
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    
    # 2. Perform Viola-Jones Object Detection
    faces = model.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=1, flags=0, minSize=(10,10), maxSize=(300,300))
    
    # 3. Print number of Faces found
    print("# faces detected:", len(faces))
    
    # 4. Draw box around faces found
    for i in range(0, len(faces)):
        predictions_set.add((faces[i][0], faces[i][1], faces[i][2], faces[i][3])) 

    return predictions_set

def readGroundtruth(filename, file):
    groundTruth_set = set()

    # read bounding boxes as ground truth
    with open(filename) as f:

        # read each line in text file
        for line in f.readlines():
            content_list = line.split(",")
            img_name = content_list[0]
            x = int(float(content_list[1]))
            y = int(float(content_list[2]))
            width = int(float(content_list[3]))
            height = int(float(content_list[4]))

            if file == img_name:
               print("ground_truth", str(x)+' '+str(y)+' '+str(width)+' '+str(height))
               groundTruth_set.add((x, y, width, height))

    return groundTruth_set

def display(frame, seto, colour):
    newImage = np.array(frame)
    for (x, y, width, height) in seto:
        start_point = (x, y)
        end_point = (x + width, y + height)
        thickness = 2

        newImage = cv2.rectangle(newImage, start_point, end_point, colour, thickness)

    return newImage
               
def assess(groundTruth_set, predictions_set):
    # There are 2 approaches here:
    # a. For each ground truth box, find the predicted box with highest IoU
    # b. For each predicted box, find the ground truth box with highest IoU
    # c. Balance it so that each ground truth can be assigned a ground truth - for each ground truth box, find the predicted box with lowest IoU

    seen = set()
    for (gx, gy, gWidth, gHeight) in groundTruth_set:
        worstFit, worstFit_iou = None, 1
        for (px, py, pWidth, pHeight) in predictions_set:
            if (px, py) in seen:
                continue
            
            # Intersection-Over-Union
            ground_truth = [gx, gy, gWidth, gHeight]
            prediction = [px, py, pWidth, pHeight]

            iouVal = iou(ground_truth, prediction)
            if 0.5 < iouVal and worstFit_iou > iouVal:        
                worstFit = (px, py, pWidth, pHeight)
                worstFit_iou = iouVal

        # Found the worst box, remove it
        if worstFit:
            seen.add(worstFit)
            print(worstFit_iou)

    # Ground Truth Boxes which have been allocated -> True Positives
    # Ground Truth Boxes which have not been allocated -> False Negatives
    # Predicted Boxes which have not been allocated -> False Positives
    TP = len(seen)
    FN = len(groundTruth_set) - len(seen)
    FP = len(predictions_set) - len(seen)
    print("True Positives:", TP)
    print("False Negatives:", FN)
    print("False Positives:", FP)

    TPR = TP / (TP + FN) if TP + FN != 0 else 0
    F_1 = TP / (TP + 1/2 * (FP + FN)) if (TP + 1/2 * (FP + FN)) != 0 else 0 
    print("True Positive Rate:", TPR)
    print("F_1 score:", F_1)


def loadClassifier(cascade_name):
    # 2. Load the Strong Classifier in a structure called `Cascade'
    model = cv2.CascadeClassifier()
    if not model.load(cv2.samples.findFile(cascade_name)):
        print('--(!)Error loading cascade model')
        exit(0)

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='no entry sign detection')
    parser.add_argument('-name', '-n', type=str, default='images/face2.jpg')
    args = parser.parse_args()

    imageName = args.name
    cascade_name = "frontalface.xml"

    # ignore if no such file is present.
    if (not os.path.isfile(imageName)) or (not os.path.isfile(cascade_name)):
        print('No such file')
        sys.exit(1)

    # 1. Read Input Image
    frame = cv2.imread(imageName, 1)

    # ignore if image is not array.
    if not (type(frame) is np.ndarray):
        print('Not image data')
        sys.exit(1)

    model = loadClassifier(cascade_name)
    groundTruth_set = readGroundtruth('groundtruth.txt')
    predictions_set = detect(frame, model)
    frame = display(frame, groundTruth_set, (0, 0, 255))
    frame = display(frame, predictions_set, (0, 255, 0))
    assess(groundTruth_set, predictions_set)

    # 4. Save Result Image
    cv2.imwrite( "detected.jpg", frame )

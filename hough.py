import numpy as np
import cv2
import os
import sys
import argparse
import math

# LOADING THE IMAGE
# Example usage: python houghCircle.py -n coins.png


# ==================================================
def convolution(input, kernel):
	kernelRadiusX = round(( kernel.shape[0] - 1 ) / 2)
	kernelRadiusY = round(( kernel.shape[1] - 1 ) / 2)

	# intialise the output using the input
	output = np.zeros([input.shape[0], input.shape[1]], dtype=np.float32)

	paddedInput = cv2.copyMakeBorder(input, 
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY, 
		cv2.BORDER_REPLICATE)

	# now we can do the convolution
	for i in range(0, input.shape[0]):	
		for j in range(0, input.shape[1]):
			patch = paddedInput[i:i+kernel.shape[0], j:j+kernel.shape[1]]
			sum = (np.multiply(patch, kernel)).sum()
			output[i, j] = sum

	return output

def gaussianBlur(input):
	size = 3
	kernel = np.ones((size, size))
	kernel[:,:] = 1/(size*size)
	return convolution(input, kernel)

def convolution(image, kernel):
    padding = kernel.shape[0] // 2
    # add padding
    for i in range(padding):
        image = np.c_[np.zeros(image.shape[0]), image[:], np.zeros(image.shape[0])]
        image = np.vstack((np.zeros(image.shape[1]), image, np.zeros(image.shape[1])))
    
    newImage = []

    for y in range(padding, image.shape[0]-padding):
        tempArray = []
        for x in range(padding, image.shape[1]-padding):
            temp = 0
            for i in range(-padding, padding+1):
                for j in range(-padding, padding+1):
                    temp += image[y-i][x-j] * kernel[padding+i][padding+j]
            tempArray.append(temp)
        newImage.append(tempArray)

    return np.array(newImage)

def sobelEdge_x(input):
	kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
	return convolution(input, kernel)

def sobelEdge_y(input):
	kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
	return convolution(input, kernel)

def sobelEdge_magnitude(input):
    sobel_x = sobelEdge_x(input)
    sobel_y = sobelEdge_y(input)
    
    sobel_x = sobel_x ** 2
    sobel_y = sobel_y ** 2
    output = np.sqrt(sobel_x + sobel_y)
    normalisedOutput = ((output - output.min()) / (output.max()-output.min()) *255).astype(np.uint8)

    # threshold = 50
    # normalisedOutput[:] = (normalisedOutput[:] > threshold) * 255
	
    return normalisedOutput

def sobelEdge_direction(input):
	sobel_x = sobelEdge_x(input)
	sobel_y = sobelEdge_y(input)

	output = sobel_y / (sobel_x + 1e-8)
	output = np.arctan(output)
	# output *= 255 / output.max()
	return output

def houghCircle(input, threshold_mag, radMin, radMax):
	sobel_mag = sobelEdge_magnitude(input)
	sobel_dir = sobelEdge_direction(input)
	hough = np.zeros([input.shape[0], input.shape[1], radMax], dtype=np.float32)

	for y in range(input.shape[0]):
		for x in range(input.shape[1]):
			
			if sobel_mag[y, x] > threshold_mag:
				for rad in range(radMin, radMax):

					x_0 = x + int(rad * np.cos(sobel_dir[y, x]))
					y_0 = y + int(rad * np.sin(sobel_dir[y, x]))
					if 0 <= x_0 and x_0 < input.shape[1] and 0 <= y_0 and y_0 < input.shape[0]:
						hough[y_0, x_0, rad] += 1

					x_0 = x - int(rad * np.cos(sobel_dir[y, x]))
					y_0 = y - int(rad * np.sin(sobel_dir[y, x]))
					if 0 <= x_0 and x_0 < input.shape[1] and 0 <= y_0 and y_0 < input.shape[0]:
						hough[y_0, x_0, rad] += 1

	return hough


# Draw a circle at any point where the number in the hough space is larger than some threshold
def displayHoughCircles(image, houghspace, threshold):
    newImage = np.array(image)
    for y in range(houghspace.shape[0]):
        for x in range(houghspace.shape[1]):
            for r in range(houghspace.shape[2]):
                if(houghspace[y][x][r] > threshold):
                    cv2.circle(newImage, (x,y), r, (0,255,0))

    return newImage

def houghLine(input, threshold_mag, delta):
	sobel_mag = sobelEdge_magnitude(input)
	sobel_dir = sobelEdge_direction(input)
	hough = np.zeros([int(np.ceil(np.sqrt(input.shape[0]**2 + input.shape[1]**2))) + 1, 360+1], dtype=np.float32) # ?
	#hough = {}


	for y in range(input.shape[0]):
		for x in range(input.shape[1]):
			if sobel_mag[y, x] > threshold_mag:
				for degree in range(0, 360+1):
					theta = np.radians(degree)

					if sobel_dir[y, x] - delta <= theta and theta <= sobel_dir[y, x] + delta:
						rho = x * np.cos(theta) + y * np.sin(theta)
						
						# if rho not in hough:
						# 	hough[rho] = {}
						# if theta not in hough[rho]:
						# 	hough[rho][theta] = 0

						#hough[rho][theta] += 1
						hough[round(rho), round(np.degrees(theta))] += 1
	
	return hough


def displayHoughLines(image, houghspace, threshold):
	newImage = np.array(image)

	for rho in range(houghspace.shape[0]):
		for theta in range(houghspace.shape[1]):
			if(houghspace[rho][theta] > threshold):
				print(theta)
				theta = np.radians(theta)
				a = np.cos(theta)
				b = np.sin(theta)
				c = -rho

				# if vertical
				if round(np.sin(theta)) == 0:
					x_intercept = round(-c / a)
					pt1 = (x_intercept, 0)
					pt2 = (x_intercept, newImage.shape[0]-1)
					
				# if horizontal
				elif round(np.cos(theta)) == 0:
					y_intercept = round(-c / b)

					pt1 = (0, y_intercept) 
					pt2 = (newImage.shape[1]-1, y_intercept)

				# all other cases
				else:
					m = (-1) * np.cos(theta) / np.sin(theta)
					c = rho / np.sin(theta)			
					pt1 = (0, round(0 * m + c))
					pt2 = (newImage.shape[1]-1, round((newImage.shape[1]-1) * m + c))

				newImage = cv2.line(newImage, pt1, pt2, (0, 255, 0), 1)

	return newImage

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='hough circle detection')
	parser.add_argument('-name', '-n', type=str, default='coins1.png')
	args = parser.parse_args()

	imageName = args.name

	# ignore if no such file is present.
	if not os.path.isfile(imageName):
		print('No such file')
		sys.exit(1)

	# Read image from file
	image = cv2.imread(imageName, 1)

	# ignore if image is not array.
	if not (type(image) is np.ndarray):
		print('Not image data')
		sys.exit(1)

	# grayscale
	if image.shape[2] >= 3:
		gray_image = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )
		gray_image = gray_image.astype(np.float32)
	else:
		gray_image = image.astype(np.float32)


	output = houghCircle(gray_image, 50, 40, 80)
	circles = displayHoughCircles(image, output, 10)
	cv2.imwrite( "output.png", circles )

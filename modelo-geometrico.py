import math
import imutils
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from image_utils import draw_lines, draw_lines_polar


# threshold
def threshold(img, th):
	a, b = img.shape[0], img.shape[1]
	for r in range(a):
		for c in range(b):
			if img[r, c] > th:
				img[r, c] = 255
			else:
				img[r, c] = 0


# transformada de hough
def hough_transform(img, theta_num=4, acc=0.005):
	x, y = img.shape
	hyp = int(math.hypot(x, y))
	accumulator = np.zeros((2 * hyp + 1, 180 * theta_num))
	pii = np.pi / (180 * theta_num)
	for row in range(x):
		for column in range(y):
			if img[row, column] > 128:
				for angle in range(180 * theta_num):
					radius = int(column * np.cos(angle * pii) + row * np.sin(angle * pii)) + hyp + 1
					accumulator[radius, angle] += acc
	return accumulator


# Toma los puntos de acumulación del espacio de hough
def get_accumulation_points(img):
	img = np.uint8(img)
	contours = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	contours_grab = imutils.grab_contours(contours)
	points_hough = []
	for contour in contours_grab:
		p = cv.moments(contour)
		if p["m00"] != 0:
			x = int(p["m10"] / p["m00"])
			y = int(p["m01"] / p["m00"])
		else:
			x, y = 0, 0
		points_hough.append([x, y])
	return points_hough


# Convierte los puntos del espacio hough en rectas
def hough_to_lines(img, points_hough, theta_num=4):
	x, y, z = img.shape
	hyp = int(math.hypot(x, y))
	pii = np.pi / (180 * theta_num)
	for point in points_hough:
		t, p = point[0], point[1]
		cos_theta = np.cos(t * pii)
		sin_theta = np.sin(t * pii)
		for row in range(x):
			for column in range(y):
				radius = int(column * cos_theta + row * sin_theta) + hyp
				if radius == p:
					img[row, column] = (0, 0, 255)

	# Intento de marcar la interseccion entre las lineas, pero no funciona
	# x, y, z = img.shape
	# hyp = int(math.hypot(x, y))
	# pii = np.pi / (180 * theta_num)
	# len_ph = len(points_hough)
	# for n in range(len_ph):
	# 	point = points_hough[n]
	# 	t, p = point[0], point[1]
	# 	cos_theta = np.cos(t * pii)
	# 	sin_theta = np.sin(t * pii)
	# 	for row in range(x):
	# 		for column in range(y):
	# 			radius = int(column * cos_theta + row * sin_theta) + hyp
	# 			if radius == p:
	# 				img[row, column] = (255, 0, 0)
	# 				for nn in range(len_ph - n - 1):
	# 					print(len_ph - nn - 1)
	# 					pointn = points_hough[len_ph - nn - 1]
	# 					tn, pn = pointn[0], pointn[1]
	# 					cos_thetan = np.cos(tn * pii)
	# 					sin_thetan = np.sin(tn * pii)
	# 					radiusn = int(column * cos_thetan + row * sin_thetan) + hyp
	# 					if radiusn == radius:
	# 						img[row, column] = (0, 0, 255)


def print_lines(img, lines):
	x, y = img.shape[0], img.shape[1]
	img_lines = np.zeros((x, y))
	for line in lines:
		for point in line:
			img_lines[point] = 255
	return img_lines


def drawpoints(img, points):
	img = np.float32(img)
	img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
	for point in points:
		cv.circle(img, point, 3, (0, 200, 255), -1)
	cv.imshow("Accumulation points (hough space)", img)


def edge_detection(img):
	# # Vertical lines
	# filter_edge = np.array([-1, 0, 1]).reshape(1, 3)
	# img_edge_v = cv.filter2D(gray, -1, filter_edge)
	# cv.imshow("vertical", img_edge_v)
	# # Horizontal lines
	# filter_edge_t = filter_edge.T
	# img_edge_h = cv.filter2D(gray, -1, filter_edge_t)
	# cv.imshow("horizontal", img_edge_h)
	# intensitity_image = np.sqrt(np.power(img_edge_h, 2) + np.power(img_edge_v, 2))
	pass


def main():
	original = cv.imread('images/football.png')
	gray = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
	# Para imagenes en colores es necesario usar el detector de bordes
	# edges = edge_detection(gray)
	threshold(gray, 128)
	accumulator = hough_transform(gray)
	cv.imshow("Accumulator (Hough space)", accumulator)
	# El segundo parametro es el treshold, bajandolo se pueden detectar lineas mas cortas
	threshold(accumulator, 0.5)
	points_hough = get_accumulation_points(accumulator)
	print(points_hough)
	hough_to_lines(original, points_hough, 4)
	drawpoints(accumulator, points_hough)
	cv.imshow("Lines", original)

	cv.waitKey(0)
	cv.destroyAllWindows()


if __name__ == '__main__':
	main()

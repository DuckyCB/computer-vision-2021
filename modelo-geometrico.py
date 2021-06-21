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
# Toma figuras (manchas) de la imagen y toma el centro de cada una
def hough_points(img):
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
	len_ph = len(points_hough)
	vertices = []
	for n in range(len_ph):
		point = points_hough[n]
		t, p = point[0], point[1]
		cos_theta = np.cos(t * pii)
		sin_theta = np.sin(t * pii)
		for row in range(x):
			for column in range(y):
				radius = int(column * cos_theta + row * sin_theta) + hyp
				if radius == p:
					img[row, column] = (255, 0, 0)
					# Para cada punto de hough, prueba todos los demás puntos que tengan rectas que se toquen.
					for nn in range(len_ph - n - 1):
						pointn = points_hough[len_ph - nn - 1]
						tn, pn = pointn[0], pointn[1]
						cos_thetan = np.cos(tn * pii)
						sin_thetan = np.sin(tn * pii)
						radiusn = int(column * cos_thetan + row * sin_thetan) + hyp
						if radiusn == pn:
							vertices.append([column, row])
	return vertices


# Dibuja puntos rojos en la imagen
def draw_points(img, points):
	for point in points:
		cv.circle(img, point, 3, (0, 0, 255), -1)
	return img


#
def edge_detection(gray):
	# Vertical lines
	filter_edge = np.array([-1, 0, 1]).reshape(1, 3)
	img_edge_v = cv.filter2D(gray, -1, filter_edge)
	threshold(img_edge_v, 60)
	cv.imshow("vertical", img_edge_v)
	# Horizontal lines
	filter_edge_t = filter_edge.T
	img_edge_h = cv.filter2D(gray, -1, filter_edge_t)
	threshold(img_edge_h, 60)
	cv.imshow("horizontal", img_edge_h)
	img_edge = img_edge_v.copy()
	x, y = img_edge_v.shape
	# Crea una imagen sumando todos los puntos de borde verticales y horizontales
	for row in range(x):
		for column in range(y):
			if img_edge_h[row, column] > 128:
				img_edge[row, column] = 255
	cv.imshow("edges", img_edge)
	return img_edge


def main():
	# Abre una imagen y detecta los bordes
	original = cv.imread('images/football.png')
	gray = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
	edges = edge_detection(gray)

	# Realiza la transformada de hough y toma los puntos con mayor acumulacion de curvas
	accumulator = hough_transform(edges)
	cv.imshow("Accumulator (Hough space)", accumulator)
	# El segundo parametro es el treshold, bajandolo se pueden detectar lineas mas cortas
	threshold(accumulator, 0.5)
	points_hough = hough_points(accumulator)
	vertices = hough_to_lines(original, points_hough, 4)
	original = draw_points(original, vertices)
	# Puntos donde se intersectan las rectas
	print(vertices)

	# Convierte el accumulator en una imagen a color para mostrar los puntos tomados
	accumulator = np.float32(accumulator)
	accumulator = cv.cvtColor(accumulator, cv.COLOR_GRAY2BGR)
	accumulator = draw_points(accumulator, points_hough)

	# Muestra la imagen final con las rectas y los puntos detectados
	cv.imshow("Lines", original)
	cv.imshow("Accumulation points (hough space)", accumulator)

	cv.waitKey(0)
	cv.destroyAllWindows()


if __name__ == '__main__':
	main()

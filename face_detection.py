import os
import pickle
from enum import Enum
from glob import glob

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import progressbar
import sklearn.neighbors
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from sklearn import svm

from evaluation import evaluate_detector, precision_and_recall, interpolated_average_precision

import image_utils


class FeatureExtractors(Enum):
	MiniImage = 1
	HOG = 2
	LBP = 3


def extract_features(method, img):
	# Switch between Feature extraction Methods
	image_representation = []

	if method == FeatureExtractors.MiniImage:
		image_representation = extract_mini_image_features(img)
	elif method == FeatureExtractors.HOG:
		image_representation = extract_hog_features(img)
	elif method == FeatureExtractors.LBP:
		image_representation = extract_lbp_features(img)

	return image_representation


def extract_mini_image_features(img, resize_size=(64, 64)):
	resized_image = cv.resize(img, resize_size)
	image_representation = resized_image.reshape(resize_size[0] * resize_size[1])
	return image_representation


def extract_lbp_features(img):
	meth = 'uniform'
	rad = 3
	n_point = 8 * rad
	lbp_img = local_binary_pattern(img, n_point, rad, meth)
	to_return = np.concatenate(lbp_img, axis=0)
	return to_return


def extract_hog_features(img):
	fd, _ = hog(img, orientations=10, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
	return fd


# Function for loading loading training data from positive and negative examples
def load_training_data(training_positive_dir, trainign_negative_dir, feature_extractor=FeatureExtractors.MiniImage):
	positive_img_files = sorted(glob(training_positive_dir + '/*'))
	negative_img_files = sorted(glob(trainign_negative_dir + '/*'))
	# comment these lines for loading all data:
	# change these lines for increasing the amount of data:

	# positive_img_files = positive_img_files[:100]
	# negative_img_files = negative_img_files[:200]

	training_dat = []
	training_label = []

	print('\n Loading positive face images')
	bar_progres = 0
	bar_positive = progressbar.ProgressBar(max_value=len(positive_img_files))
	bar_positive.update(bar_progres)

	for img in positive_img_files:
		gray = cv.imread(img, cv.IMREAD_GRAYSCALE)
		image_representation = extract_features(feature_extractor, gray)
		training_dat.append(image_representation)
		training_label.append(1)

		bar_progres += 1
		bar_positive.update(bar_progres)

	print('\n Loading negative face images')
	bar_progres = 0
	bar_negative = progressbar.ProgressBar(max_value=len(negative_img_files))
	bar_negative.update(bar_progres)

	for img in negative_img_files:
		gray = cv.imread(img, cv.IMREAD_GRAYSCALE)
		image_representation = extract_features(feature_extractor, gray)
		training_dat.append(image_representation)
		training_label.append(0)

		bar_progres += 1
		bar_negative.update(bar_progres)

	training_dat = np.asarray(training_dat)
	training_label = np.asarray(training_label)
	return training_dat, training_label


def load_validation_data(validation_data_dir):
	validation_image_files = sorted(glob(validation_data_dir + '/*'))
	val_images = []
	for img_file in validation_image_files:
		img = cv.imread(img_file, cv.IMREAD_COLOR)
		val_images.append(img)

	return val_images


def sliding_window_full(img, window_siz, scale, stride):
	window_rows = window_siz[0]
	window_cols = window_siz[1]

	r = [i for i in range(0, 186, 32)]  # [0, 5, 10, 15, ... , 180, 185] Punto de arriba a la izquierda
	c = [i for i in range(0, 186, 32)]  # [0, 5, 10, 15, ... , 180, 185]

	pats = np.zeros((window_rows, window_cols, len(r) * len(c)))
	bbox_locs = np.zeros((len(r) * len(c), 4))

	t = 0
	for i in range(0, len(r)):
		for j in range(0, len(c)):
			pats[:, :, t] = img[r[i]:r[i] + window_rows, c[j]:c[j] + window_cols]
			bbox_locs[t, :] = [r[i], c[j], window_rows, window_cols]  # top-left y,x, height, width
			t += 1

	return pats, bbox_locs


def get_patches(gray_img, w_size, classif):
	pats, bbox_locs = sliding_window_full(gray_img, w_size, 1, 32)
	# You need to extract features for every patch (same features you used for training the classifier)
	patches_feature_rep = []
	for i in range(pats.shape[2]):
		patch_representation = extract_features(FeatureExtractors.HOG, pats[:, :, i])
		patches_feature_rep.append(patch_representation)
	patches_feature_rep = np.asarray(patches_feature_rep)
	# Get score for each sliding window patch
	score = classif.predict_proba(patches_feature_rep)

	return patches_feature_rep, pats, bbox_locs, score


if __name__ == '__main__':

	input_image = 'data/face_detection/val_face_detection_images/seen_Eric_Bana_0001.jpg'
	gray_eric = cv.imread(input_image, cv.IMREAD_GRAYSCALE)
	window_size = [64, 64]
	# window_size = [84, 84]
	patches, bbox_locations = sliding_window_full(gray_eric, window_size, 1, 32)
	image_utils.show_image_with_bbox(gray_eric, bbox_locations)

	data_dir = 'data'
	face_detection_dir = os.path.join(data_dir, 'face_detection')
	training_faces_dir = os.path.join(face_detection_dir, 'cropped_faces')
	negative_examples_training_dir = os.path.join(face_detection_dir, 'non_faces_images', 'neg_cropped_img')
	validation_faces_dir = os.path.join(face_detection_dir, 'val_face_detection_images')
	validation_raw_faces_dir = os.path.join(face_detection_dir, 'val_raw_images')

	training_data, training_labels = \
		load_training_data(training_faces_dir, negative_examples_training_dir, FeatureExtractors.HOG)

	validation_data = load_validation_data(validation_faces_dir)
	knn_classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors=8)
	knn_classifier.fit(training_data, training_labels)

	# Usar esta variable en True para usar SVM
	use_svm = False
	if use_svm:
		clf = svm.SVC(C=5.0, probability=True)
		clf.fit(training_data, training_labels)
		pickle.dump(clf, open('./face_detector', 'wb'))
	else:
		pickle.dump(knn_classifier, open('./face_detector', 'wb'))

	classifier = pickle.load(open('./face_detector', 'rb'))

	# bar_progress = 0
	# bar = progressbar.ProgressBar(max_value=len(validation_data))
	#
	# window_size = [64, 64]
	# predictions = []
	# threshold_p = 0.8
	# overlap_threshold = 0.5
	# validation_data = load_validation_data(validation_faces_dir)
	# bar.update(bar_progress)
	# for image_valid in validation_data:
	# 	gray_image = cv.cvtColor(image_valid, cv.COLOR_RGB2GRAY)
	# 	# patches, bbox_locations = sliding_window(gray_image,window_size,1,32)
	# 	patches_feature_representation, patches, bbox_locations, scores = get_patches(gray_image, window_size,
	# 																				  classifier)
	# 	# Get prediction label for each sliding window patch
	# 	labels = classifier.predict(patches_feature_representation)
	# 	# Positive Face Probabilities
	# 	face_probabilities = scores[:, 1]
	# 	face_bboxes = bbox_locations[face_probabilities > threshold_p]
	# 	face_bboxes_probabilites = face_probabilities[face_probabilities > threshold_p]
	# 	# Do non max suppression and select strongest probability box
	# 	[selected_bbox, selected_score] = non_max_suppression(face_bboxes, face_bboxes_probabilites, 0.3)
	# 	show_image_with_bbox(image_valid, selected_bbox)
	#
	# 	bar_progress += 1
	# 	bar.update(bar_progress)

	# Cuenta la cantidad de imagenes que hay para analizar
	qty = 0
	for subject_folder in sorted(glob(validation_raw_faces_dir + '/*')):
		for imag in sorted(glob(subject_folder + '/*.jpg')):
			qty += 1

	bar_progress = 0
	bar = progressbar.ProgressBar(max_value=qty)

	total_true_positives = []
	total_real_positives = []
	total_positive_predictions = []
	window_size = [64, 64]
	bar.update(bar_progress)
	for subject_folder in sorted(glob(validation_raw_faces_dir + '/*')):
		for imag in sorted(glob(subject_folder + '/*.jpg')):
			gray_image = cv.imread(imag, cv.IMREAD_GRAYSCALE)
			patches_feature_representation, patches, bbox_locations, scores = \
				get_patches(gray_image, window_size, classifier)
			# Positive Face Probabilities
			face_probabilities = scores[:, 1]

			[detected_true_positives, image_real_positives, detected_faces] = \
				evaluate_detector(bbox_locations, face_probabilities)
			total_true_positives.append(detected_true_positives)
			total_real_positives.append(image_real_positives)
			total_positive_predictions.append(detected_faces)

			bar_progress += 1
			bar.update(bar_progress)

	total_true_positives = np.asarray(total_true_positives)
	total_real_positives = np.asarray(total_real_positives)
	total_positive_predictions = np.asarray(total_positive_predictions)

	precision, recall = precision_and_recall(total_true_positives, total_real_positives, total_positive_predictions)

	plt.plot(recall, precision)
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.xlim(0, 1.1)
	plt.ylim(0, 1.1)

	ap = interpolated_average_precision(recall, precision)

	print('\nDetection Average Precision is {}'.format(ap))

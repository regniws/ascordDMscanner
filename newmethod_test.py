import datamatrix_locator as ddl
import cv2
import math
import numpy as np

from datamatrix_locator import DatamatrixLocator
from point import Point
from color import Color

# image = cv2.imread('qr5_roi.png')
# image = cv2.imread('qr_test.png')
image = cv2.imread('../algo_backup/pack/photo_2023-10-03_11-08-13.jpg')

# alg = ddm.DottedDataMatrix()
# alg.detect_datamatrix(image, True, True)

alg = DatamatrixLocator()

# Convert to a grayscale image
image_mono = alg.rgb2gray(image)

# Gaussian smoothing
k = 7
sigmaX =3
gauss = alg.gaussian_smoothing(image_mono, k, sigmaX)

# Dynamic mean filter
mean_filtered = alg.dynamic_mean_filter(image_mono, k)

# block_size = 35
# C = 16
block_size = 21
C = 7

# image_threshold = alg.bernsenThreshold(image_mono, block_size, C)
# image_threshold = alg.kittler_threshold(image_mono, 3, 1, 1)
image_threshold = alg.adaptive_threshold(mean_filtered, block_size, C)

# Perform morphological close
close_size = 2
image_morphed = alg.close_morph(image_threshold, close_size)

# Find a bunch of contours in the image.
contours = alg.get_contours(image_morphed)
polygons = alg.contours_to_polygons(contours)

# Convert lists of vertices to lists of edges (easier to work with).
edge_sets1 = map(alg.polygons_to_edges, polygons)

# Discard all edge sets which probably aren't datamatrix perimeters.
edge_sets2 = list(filter(alg._filter_non_trivial, edge_sets1))
edge_sets3 = list(filter(alg._filter_longest_adjacent, edge_sets2))
edge_sets4 = list(filter(alg._filter_longest_approx_orthogonal, edge_sets3))
edge_sets5 = list(filter(alg._filter_longest_similar_in_length, edge_sets4))

# Convert edge sets to FinderPattern objects
fps = [alg.get_finder_pattern(es) for es in edge_sets4]

def polygon_image(lines):
    blank = np.zeros(image.shape, np.uint8)
    img = cv2.drawContours(blank, lines, -1, (0, 255, 0), 1)
    return img


def edges_image(edge_sets):
    blank = np.zeros(image.shape, np.uint8)

    for shape in edge_sets:
        for edge in shape:
            print(edge[1])
            cv2.line(blank, Point.from_array(edge[0]).tuple(),
                            Point.from_array(edge[1]).tuple(), Color.Green().rgb(), 1)

    return blank

# Make images
image_contours = polygon_image(contours)
image_polygons = polygon_image(polygons)
image_filter1 = edges_image(edge_sets2)
image_filter2 = edges_image(edge_sets5)

# Make finder pattern image
image_fp = np.zeros(image.shape, np.uint8)
# fps[0].draw_to_image(image_fp)

row1 = np.concatenate((image, image_fp), axis=1)
row2 = np.concatenate((image_threshold, image_morphed), axis=1)
row3 = np.concatenate((image_contours, image_polygons), axis=1)
row4 = np.concatenate((image_filter1, image_filter2), axis=1)
cv2.imshow('Row1', row1)
cv2.waitKey(0)

cv2.imshow('Row2', row2)
cv2.waitKey(0)

cv2.imshow('Row3', row3)
cv2.waitKey(0)

cv2.imshow('Row4', row4)
cv2.waitKey(0)

cv2.imshow('Image GRay', image_mono)
cv2.waitKey(0)
cv2.destroyAllWindows()

# block_size = 7
# c = 5
# close_size = 3
#
# alg = ddl.DatamatrixLocator()
#
# gray = alg.do_gray(image)
# cv2.imshow('Gray', gray)
# cv2.waitKey(0)
#
# thresholdeda = alg.do_threshold(gray, block_size, c)
# cv2.imshow('Thresholded', thresholdeda)
# cv2.waitKey(0)
#
# mean = alg.dynamic_mean_filter(thresholdeda)
# gauss = alg.gaussian_smoothing(thresholdeda)
#
# T = alg.kittler_threshold(gauss)
# _, thresholded = cv2.threshold(gauss, T, 255, cv2.THRESH_BINARY)
#
# log_image = cv2.Laplacian(thresholded, cv2.CV_32F, ksize=5)
# cv2.normalize(log_image, log_image, 3, 2, cv2.NORM_MINMAX);
# log_image = cv2.convertScaleAbs(log_image)
# # # Create the sharpening kernel
# kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
# # Sharpen the image
# log_image = cv2.filter2D(log_image, -1, kernel)
#
#
#
# thresholded2 = alg.bernsen_threshold(log_image, 5, 11)
# cv2.imshow('Thresholded2', thresholded2)
# cv2.waitKey(0)
#
# binary = alg.do_morph(thresholded2, close_size)
# cv2.imshow('Binary', binary)
# cv2.waitKey(0)
#
# contours = alg.get_contours(binary)
# contours_image = alg.polygon_image(image, contours)
# cv2.imshow('Contoured', contours_image)
# cv2.waitKey(0)
#
# # rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
# # blackhat = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, rectKernel)
# # morph = blackhat
#
# # warped = None
# # edges = cv2.Canny(gauss, 50, 250)
# # contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# #
# # # contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # # filtered_contours = contours
# # filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 10]
# #
# # # Loop through filtered contours and perform perspective transformation and decoding
# # ind = 0
# # data_matrix_coords = None
# # for contour in filtered_contours:
# #     ind = ind + 1
# #     # Perform perspective transformation
# #     rect = cv2.minAreaRect(contour)
# #     box = cv2.boxPoints(rect)
# #     box = np.int0(box)
# #
# #     warped = self.four_point_transform(morph, box)
# #     # cv2.drawContours(image, [box], -1, (0.255, 255), 2)
# #     # cv2.imshow('Tested' + str(ind), warped)
# #
# #     epsilon = 0.04 * cv2.arcLength(contour, True)
# #     approx = cv2.approxPolyDP(contour, epsilon, True)
# #
# #     # Check if the polygon has four corners (a rectangle)
# #     if len(approx) == 4:
# #         x, y, w, h = cv2.boundingRect(contour)
# #         ratio = float(w) / h
# #         if 0.9 <= ratio <= 1.1:
# #             data_matrix_coords = approx
# #             # break
# #
# #     # break
# #
# # if data_matrix_coords is not None:
# #     # Perform perspective transformation to align the dots
# #     warped = self.four_point_transform(morph, data_matrix_coords.reshape(4, 2))
# # output = warped
# # # output = self.dott_detection(th["threshold"], image)
# # # output = self.dott_detection(edges, image)
# #
# # # roi = self.get_roi(image)
# # # roi = self.four_point_transform(binary, roi[1].reshape(4, 2))
#
#

import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
from pylibdmtx.pylibdmtx import decode

class DottedDataMatrix:
    DEFAULT_SIZE = 14
    DEFAULT_SIDE_SIES = [12, 14]

    def __init__(self):
        self._matrix_sizes = [self.DEFAULT_SIZE]

        self._data = None
        self._error_message = ""
        self._valid = False

    def set_matrix_sizes(self, matrix_sizes):
        self._matrix_sizes = [int(v) for v in matrix_sizes]

    def is_valid(self):
        return self._valid

    def data(self):
        if not self._is_readed:
            return ""

        return self._data

    def bounds(self):
        return self._bounds

    def center(self):
        return self._center

    def radius(self):
        return self._radius

    def get_roi(self, input):
        image = input
        gray = self.gray_processing(image)
        gauss = self.gaussian_smoothing(gray)
        mean_filt = self.dynamic_mean_filter(gray)

        edges = cv2.Canny(mean_filt, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out small contours
        # filtered_contours = contours
        filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 100]
        # Loop through filtered contours and perform perspective transformation and decoding
        ind = 0
        data_matrix_coords = None
        for contour in filtered_contours:
            ind = ind + 1
            # Perform perspective transformation
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            warped = self.four_point_transform(image, box)
            data_matrix_coords = box
            # tt1 = cv2.drawContours(image, [box], -1, (0.255, 255), 2)
            # cv2.imshow('Tested' + str(ind), warped)

            # Approximate the contour to a polygon
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if the polygon has four corners (a rectangle)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                ratio = float(w) / h
                if ratio >= 0.9 and ratio <= 1.1:
                    data_matrix_coords = approx
                    break
            break

        if data_matrix_coords is not None:
            # Perform perspective transformation to align the dots
            warped = self.four_point_transform(image, data_matrix_coords.reshape(4, 2))

        return [warped, data_matrix_coords]

    def detect_datamatrix(self, input, show_detection=False, show_analysis=False):
        image = input
        roi_info = self.get_roi(image)
        box = roi_info[1].reshape(4,2)
        left = min(box[:][0])
        right = max(box[:][0])
        top = min(box[:][1])
        bottom = max(box[:][1])

        points = [(left, top), (right, bottom)]

        roi = self.four_point_transform(image, roi_info[1].reshape(4, 2))
        if show_detection:
            cv2.drawContours(input,
                    [roi_info[1].reshape(4, 2)],
                               -1, (0.255, 255),
                               2)
        # image = roi

        gray = self.gray_processing(image)

        # thresh = cv2.adaptiveThreshold(gray, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
        # gray = thresh
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

        gauss = self.gaussian_smoothing(gray)

        mean = self.dynamic_mean_filter(gray)

        T = self.kittler_threshold(gauss)
        _,thresholded = cv2.threshold(gauss, T, 255, cv2.THRESH_BINARY)

        log_image = cv2.Laplacian(thresholded, cv2.CV_32F, ksize=3)
        cv2.normalize(log_image, log_image, 1, 0, cv2.NORM_MINMAX);
        log_image = cv2.convertScaleAbs(log_image)
        # # Create the sharpening kernel
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        # Sharpen the image
        log_image = cv2.filter2D(log_image, -1, kernel)


        binary = self.bernsen_threshold(mean, 3, 45)
        morph = self.morphology(binary)


        morph = cv2.adaptiveThreshold(morph, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)

        warped = None
        edges = cv2.Canny(gauss, 50, 250)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # filtered_contours = contours
        filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 10]

        # Loop through filtered contours and perform perspective transformation and decoding
        ind = 0
        data_matrix_coords = None
        for contour in filtered_contours:
            ind = ind + 1
            # Perform perspective transformation
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            warped = self.four_point_transform(morph, box)
            # cv2.drawContours(image, [box], -1, (0.255, 255), 2)
            # cv2.imshow('Tested' + str(ind), warped)

            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if the polygon has four corners (a rectangle)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                ratio = float(w) / h
                if 0.9 <= ratio <= 1.1:
                    data_matrix_coords = approx
                    # break

            # break


        if data_matrix_coords is not None:
            # Perform perspective transformation to align the dots
            warped = self.four_point_transform(morph, data_matrix_coords.reshape(4, 2))
        output = warped
        # output = self.dott_detection(th["threshold"], image)
        # output = self.dott_detection(edges, image)

        # roi = self.get_roi(image)
        # roi = self.four_point_transform(binary, roi[1].reshape(4, 2))


        decoded_info = decode(gauss, 70, shape=2)
        print("Decoded data")
        data_matrix_text = None

        if decoded_info:
            data_matrix_text = decoded_info[0].data.decode('utf-8')
            print("Decoded Data Matrix:", data_matrix_text)
        else:
            print("Data Matrix not decoded.")
        if show_analysis:
            self.show_pipeline(input, gray, gauss,
                               thresholded, log_image, mean,
                               binary, morph, roi)
        return [points, data_matrix_text]

    def show_pipeline(self,
                      input, gray, gauss,
                      thresholded, log_image, mean,
                      binary, morph, roi):
        titles = [
            'Original', 'Gray', 'Gaussian',
            'thresholded', 'log_image', 'mean',
            'Bernsen', 'Morph', 'ROI'
        ]

        images = [
            input, gray, gauss,
            thresholded, log_image, mean,
            binary, morph, roi
        ]

        img_num = len(images)
        for i in range(img_num):
            # if math.floor(i / 3) == 1 or (math.floor(i / 3) == 2 and i % 3 == 1):
            #     plt.subplot(math.ceil(img_num / 3), 3, i + 1), plt.plot(images[i], 'gray')
            #     plt.title(titles[i])
            #     plt.xticks([]), plt.yticks([])
            #     plt.xlim([0, 256])
            # else:
            plt.subplot(math.ceil(img_num / 3), 3, i + 1), plt.imshow(images[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()


    def gray_processing(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def gaussian_smoothing(self, gray):
        return cv2.GaussianBlur(gray, (3, 3), 3)

    def dynamic_mean_filter(self, image):
        return cv2.medianBlur(image, 5)

    def morph_shape(self, val):
        if val == 0:
            return cv2.MORPH_RECT
        elif val == 1:
            return cv2.MORPH_CROSS
        elif val == 2:
            return cv2.MORPH_ELLIPSE

    def local_min(self, image, erosion_size):
        erosion_shape = self.morph_shape(0)
        element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                           (erosion_size, erosion_size))
        return cv2.erode(image, element)

    def local_max(self, image, dilatation_size):
        dilation_shape = self.morph_shape(0)
        element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                           (dilatation_size, dilatation_size))
        return cv2.dilate(image, element)

    def binarization(self, gray, T1, T2, T3, T4, a):
        h = gray.shape[0]
        w = gray.shape[1]
        img1 = np.zeros((h, w), np.uint8)
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                # if gray[i,j] > T1 + a:
                #     img1[i,j] = 255
                # else:
                #     img1[i,j] = 0

                if T3[i, j] >= a:
                    img1[i, j] = (gray[i, j] >= T2[i, j]) * 255
                else:
                    img1[i, j] = (gray[i, j] >= T4[i, j]) * 255
        return img1

    def kittler_threshold(self, gray):
        # kittler: Global threshold
        gX = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        gY = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        gMax = np.maximum(np.abs(gX), np.abs(gY))
        T = (np.multiply(gMax, gray)).sum() / (gMax.sum())
        return T

    def bernsen_threshold(self, gray, w, a):

        T1 = self.kittler_threshold(gray)
        # Bernsen improved
        kernel_d = np.ones((2*w, 2*w), np.float32)
        kernel_s = np.ones((w, w), np.float32)

        minD = cv2.erode(gray, kernel_d)
        maxD = cv2.dilate(gray, kernel_d)
        minS = cv2.erode(gray, kernel_s)
        maxS = cv2.dilate(gray, kernel_s)

        T2 = (maxD + minD)/2.0
        T3 = maxS - minS
        T4 = (T1 + T2)/2.0

        binary = self.binarization(gray, T1, T2, T3, T4, a)
        return binary

    def morphology(self, binary):
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, element, iterations=1)
        closed = cv2.morphologyEx(open, cv2.MORPH_CLOSE, element, iterations=1)

        return closed

    def four_point_transform(self, image, pts):
        # Sort the points in top-left, top-right, bottom-right, and bottom-left order
        rect = np.zeros((4, 2), dtype="float32")
        s = np.sum(pts, axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        (tl, tr, br, bl) = rect

        # Calculate the width and height of the new image
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # Define the new transformed coordinates
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

        # Perform the perspective transform
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        return warped



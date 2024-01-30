import math
import cv2

import numpy as np
from functools import partial, reduce
from operator import add

from point import Point
from finderpattern import FinderPattern

OPENCV_VERSION = cv2.__version__[0]


def length(edge):
    return distance(*edge)

def distance(point_a, point_b):
    return modulus(np.subtract(point_a, point_b))

def modulus(vector):
    return quadrature_add(*vector)

def cosine(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (modulus(vec_a) * modulus(vec_b))

def quadrature_add(self, *values):
    return math.sqrt(reduce(add, (c * c for c in values)))


""" Finding datamatrix positions """
class DatamatrixLocator:

    def __init__(self):
        pass

    def rgb2gray(self, image):
        """ Converting image to gray """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def gaussian_smoothing(self, gray, ksize, sigmaX):
        return cv2.GaussianBlur(gray, (ksize, ksize), sigmaX)\

    def dynamic_mean_filter(self, image, ksize):
        return cv2.medianBlur(image, ksize)

    def bernsenThreshold(self, gray, block_size, c):
        blurred = cv2.GaussianBlur(gray, (block_size, block_size), 0)
        (T, threshInv) = cv2.threshold(blurred, 230,
                                       255, cv2.THRESH_BINARY_INV)
        cv2.imshow("Blurred thresholded image", threshInv)
        cv2.waitKey(0)


    def kittler_threshold(self, gray, ksize, scale, delta):
        # kittler: Global threshold
        ddepth = cv2.CV_64F
        # ddepth = cv2.CV_8UC1
        # ddepth = cv2.CV_32SC1

        gX = cv2.Sobel(gray, ddepth, 1, 0,
                       ksize=ksize, scale=scale,
                       delta=delta,
                       borderType=cv2.BORDER_DEFAULT)
        gY = cv2.Sobel(gray, ddepth, 0, 1,
                       ksize=ksize, scale=scale,
                       delta=delta,
                       borderType=cv2.BORDER_DEFAULT)
        abs_grad_x = cv2.convertScaleAbs(gX)
        abs_grad_y = cv2.convertScaleAbs(gY)
        # gMax = np.maximum(np.abs(gX), np.abs(gY))
        gMax = np.maximum(abs_grad_x, abs_grad_y)
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

    def adaptive_threshold(self, gray, block_size, c):
        """ Perform an adaptive(local) threshold operation on the image. """
        return cv2.adaptiveThreshold(gray, 255.0,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY,
                                     block_size, c)

    def close_morph(self, threshold_image, morph_size):
        """ Perform a generic morphological operation on an image. """
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_size, morph_size))
        closed = cv2.morphologyEx(threshold_image, cv2.MORPH_CLOSE, element, iterations=1)
        return closed

    def get_contours(self, binary_image):
        """ Find contours and return them as lists of vertices. """
        raw_img = binary_image.copy()

        # List of return values changed between version 2 and 3
        if OPENCV_VERSION == '3':
            _, raw_contours, _ = cv2.findContours(raw_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        else:
            raw_contours, _ = cv2.findContours(raw_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        return raw_contours

    def contours_to_polygons(self, contours, epsilon=6.0):
        """ Uses the Douglas-Peucker algorithm to approximate a polygon as a similar polygon with
        fewer vertices, i.e., it smooths the edges of the shape out. Epsilon is the maximum distance
        from the contour to the approximated contour; it controls how much smoothing is applied.
        A lower epsilon will mean less smoothing. """
        shapes = [cv2.approxPolyDP(rc, epsilon, True).reshape(-1, 2) for rc in contours]
        return shapes

    def pairs_circular(self, iterable):
        """ Generate pairs from an iterable. Best illustrated by example:
        # >>> list(pairs_circular('abcd'))
        [('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'a')]
        """
        iterator = iter(iterable)
        x = next(iterator)
        zeroth = x  # Keep the first element so we can wrap around at the end.
        while True:
            try:
                y = next(iterator)
                yield((x, y))
            except StopIteration:  # Iterator is exhausted so wrap around to start.
                try:
                    yield((y, zeroth))
                except UnboundLocalError:  # Iterable has one element. No pairs.
                    pass
                break
            x = y


    def polygons_to_edges(self, vertex_list):
        """ Return a list of edges based on the given list of vertices. """
        return list(self.pairs_circular(vertex_list))

    def get_shared_vertex(edge_a, edge_b):
        """ Return a vertex shared by two edges, if any.
        """
        for vertex_a in edge_a:
            for vertex_b in edge_b:
                if (vertex_a == vertex_b).all():
                    return vertex_a

    def get_other_vertex(vertex, edge):
        """ Return an element of `edge` which does not equal `vertex`.
        """
        for vertex_a in edge:
            if not (vertex_a == vertex).all():
                return vertex_a

    def longest_pair_indices(edges):
        """ Return the indices of the two longest edges in a list of edges.
        """
        lengths = list(map(length, edges))
        return np.asarray(lengths).argsort()[-2:][::-1]

    def get_finder_pattern(self, edges):
        """ This function finds the corner between the longest two edges, which should
        be spatially adjacent (it is up to the caller to make sure of this). It
        returns the position of the corner, and vectors corresponding to the said
        two edges, pointing away from the corner. These two vectors are returned in
        an order such that their cross product is positive, i.e. (see diagram) the
        base vector (a) comes before the side vector (b).

              ^side
              |
              |   base
              X--->
         corner

        This provides a convenient way to refer to the position of a datamatrix.
        """

        i, j = self.longest_pair_indices(edges)
        pair_longest_edges = [edges[x] for x in (i, j)]
        x_corner = self.get_shared_vertex(*pair_longest_edges)
        c, d = map(partial(self.get_other_vertex, x_corner), pair_longest_edges)
        vec_c, vec_d = map(partial(np.add, -x_corner), (c, d))
        if vec_c[0] * vec_d[1] - vec_c[1] * vec_d[0] < 0:
            vec_base, vec_side = vec_c, vec_d
        else:
            vec_base, vec_side = vec_d, vec_c

        x_corner = Point(x_corner[0], x_corner[1]).intify()
        vec_base = Point(vec_base[0], vec_base[1]).intify()
        vec_side = Point(vec_side[0], vec_side[1]).intify()
        return FinderPattern(x_corner, vec_base, vec_side)


    def find_datamatrices(self, image):
        """ Finding data matrices in the image """
        gray = self.rgb2gray(image)
        return self.locate_datamatrices(self, gray)

    def locate_datamatrices(self, gray, blocksize, close_size):
        """ Image preprocessing and finding location of data matrices """
        # adaptive threshold
        adaptive = self.adaptiveThreshold(gray, blocksize)

    @staticmethod
    def _filter_non_trivial(edge_set):
        """Return True iff the number of edges is non-small.
        """
        return len(edge_set) > 6

    @staticmethod
    def _filter_longest_adjacent(edges):
        """Return True iff the two longest edges are adjacent.
        """
        i, j = DatamatrixLocator.longest_pair_indices(edges)
        return abs(i - j) in (1, len(edges) - 1)

    @staticmethod
    def _filter_longest_approx_orthogonal(edges):
        """Return True iff the two longest edges are approximately orthogonal.
        """
        i, j = DatamatrixLocator.longest_pair_indices(edges)
        v_i, v_j = (np.subtract(*edges[x]) for x in (i, j))
        return abs(cosine(v_i, v_j)) < 0.1

    @staticmethod
    def _filter_longest_similar_in_length(edges):
        """Return True iff the two longest edges are similar in length.
        """
        i, j = DatamatrixLocator.longest_pair_indices(edges)
        l_i, l_j = (length(edges[x]) for x in (i, j))
        return abs(l_i - l_j)/abs(l_i + l_j) < 0.1

    @staticmethod
    def longest_pair_indices(edges):
        """Return the indices of the two longest edges in a list of edges.
        """
        lengths = list(map(length, edges))
        return np.asarray(lengths).argsort()[-2:][::-1]

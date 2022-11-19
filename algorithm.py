import cv2 as cv
import numpy as np


def mse(imageA: object, imageB: object) -> float:
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def SIFT(choice: str, img: str) -> (int, int):
    img1 = cv.imread("image.orig/examples/"+choice)
    img2 = cv.imread(img)
    if img1 is None or img2 is None:
        print('Error loading images!')
        exit(0)
    # -- Step 1: Detect the keypoints using SIFT Detector, compute the descriptors
    minHessian = 400
    detector = cv.SIFT_create()
    keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(img2, None)
    # -- Step 2: Matching descriptor vectors with a brute force matcher
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE)
    matches = matcher.match(descriptors1, descriptors2)
    # -- Draw matches
    # img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
    # cv.drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches)
    # -- Show detected matches
    # cv.imshow('Matches: SIFT (Python)', img_matches)
    # cv.waitKey()

    # draw good matches
    matches = sorted(matches, key=lambda x: x.distance)
    min_dist = matches[0].distance
    good_matches = tuple(filter(lambda x: x.distance <= 2 * min_dist, matches))

    # img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
    # cv.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches,
    #                flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # -- Show detected matches
    # cv.imshow('Good Matches: SIFT (Python)', img_matches)
    # cv.waitKey()
    return len(matches), len(good_matches)

import pickle

import cv2 as cv
import numpy as np
from glob import glob
import algorithm

# the directory of the image database
database_dir = "image.orig"


# Compute pixel-by-pixel difference and return the sum
def compareImgs(img1, img2):
    # resize img2 to img1
    img2 = cv.resize(img2, (img1.shape[1], img1.shape[0]))
    diff = cv.absdiff(img1, img2)
    return diff.sum()


def compareImgs_hist(img1, img2):
    width, height = img1.shape[1], img1.shape[0]
    img2 = cv.resize(img2, (width, height))
    num_bins = 10

    hist1 = cv.calcHist([img1], [0], None, [num_bins], [0, 255])
    hist2 = cv.calcHist([img2], [0], None, [num_bins], [0, 255])
    sum = 0
    for i in range(num_bins):
        sum += abs(hist1[i] - hist2[i])
    return sum / float(width * height)


# 3
def retrieval0(choice):
    src_input = cv.imread("image.orig/examples/" + choice)
    min_diff = 1e50

    cv.imshow("Input", src_input)

    # change the image to gray scale
    src_gray = cv.cvtColor(src_input, cv.COLOR_BGR2GRAY)

    # read image database
    database = sorted(glob(database_dir + "/*.jpg"))

    for img in database:
        # read image
        img_rgb = cv.imread(img)
        # convert to gray scale
        img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
        # compare the two images
        # diff = compareImgs(src_gray, img_gray)
        # compare the two images by histogram, uncomment the following line to use histogram
        diff = compareImgs_hist(src_gray, img_gray)
        print(img, diff)
        # find the minimum difference
        if diff <= min_diff:
            # update the minimum difference
            min_diff = diff
            # update the most similar image
            closest_img = img_rgb
            result = img

    print("the most similar image is %s, the pixel-by-pixel difference is %f " % (result, min_diff))
    print("\n")

    cv.imshow("Result", closest_img)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 1, 4, 6
def retrieval1(choice):
    src_input = cv.imread("image.orig/examples/" + choice)
    min_diff = 1e50

    cv.imshow("Input", src_input)

    # change the image to gray scale
    src_gray = cv.cvtColor(src_input, cv.COLOR_BGR2GRAY)

    # read image database
    database = sorted(glob(database_dir + "/*.jpg"))

    for img in database:
        # read image
        img_rgb = cv.imread(img)
        # convert to gray scale
        img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
        # compare the two images
        diff = compareImgs(src_gray, img_gray)
        # compare the two images by histogram, uncomment the following line to use histogram
        # diff = compareImgs_hist(src_gray, img_gray)
        print(img, diff)
        # find the minimum difference
        if diff <= min_diff:
            # update the minimum difference
            min_diff = diff
            # update the most similar image
            closest_img = img_rgb
            result = img

    print("the most similar image is %s, the pixel-by-pixel difference is %f " % (result, min_diff))
    print("\n")

    cv.imshow("Result", closest_img)
    cv.waitKey(0)
    cv.destroyAllWindows()


# SIFT
def retrieval2(choice):
    src_input = cv.imread("image.orig/examples/" + choice)
    max_match = 0

    # src_input = cv.imread("man.jpg")

    cv.imshow("Input", src_input)
    src_gray = cv.cvtColor(src_input, cv.COLOR_BGR2GRAY)

    detector = cv.SIFT_create()
    keypoints1, descriptors1 = detector.detectAndCompute(src_gray, None)
    descriptors1 /= (descriptors1.sum(axis=1, keepdims=True) + 1e-7)
    descriptors1 = np.sqrt(descriptors1)
    # read image database
    database = sorted(glob(database_dir + "/*.jpg"))
    bf = cv.BFMatcher()

    for img in database:
        # read image
        with open("descriptors/"+img[11:-3]+"pkl", 'rb') as f:
            descriptors2 = pickle.load(f)
        # compare the two images
        # -- Step 2: Matching descriptor vectors with a brute force matcher
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        # matches = sorted(matches, key=lambda x: x.distance)
        good_matches = 0
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good_matches += 1
        print(img, good_matches)
        # find the minimum difference
        if good_matches > max_match:
            # update the minimum difference
            max_match = good_matches
            # update the most similar image
            result = img

    print("the most similar image is %s, the pixel-by-pixel difference  " % result)
    print("\n")

    closest_img = cv.imread(result)
    cv.imshow("Result", closest_img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def main():
    print("1: Image retrieval demo")
    print("2: SIFT demo")
    # to simplify if clauses
    dic = {1: "beach.jpg", 2: "building.jpg", 3: "bus.jpg", 4: "dinosaur.jpg",
           5: "flower.jpg", 6: "horse.jpg", 7: "man.jpg"}
    number = int(input("Type in the number to choose a demo and type enter to confirm\n"))
    if number == 1:
        print("1: beach")
        print("2: building")
        print("3: bus")
        print("4: dinosaur")
        print("5: flower")
        print("6: horse")
        print("7: man")
        choice = eval(input("Type in the number to choose a category and type enter to confirm\n"))

        print("You choose: %s\n" % dic[choice])
        retrieval2(dic[choice])
    elif number == 2:
        algorithm.SIFT()
    # pass
    else:
        print("Invalid input")
        exit()


if __name__ == '__main__':
    main()


import pickle
from glob import glob

from scipy.spatial import distance

# the directory of the image database
database_dir = "image.orig/datasets"


# abandoned
# # Compute pixel-by-pixel difference and return the sum
# def compareImgs(img1, img2):
#     # resize img2 to img1
#     img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
#     diff = cv2.absdiff(img1, img2)
#
#     return diff.sum(), np.mean(np.absolute(img1 - img2) / img1)-0.2
#
#
# def mse(imageA, imageB) -> float:
#     # resize img2 to img1
#     imageB = cv2.resize(imageB, (imageA.shape[1], imageA.shape[0]))
#     err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
#     err /= float(imageA.shape[0] * imageA.shape[1])
#
#     # return the MSE, the lower the error, the more "similar"
#     return err
#
#
# def compareImgs_hist(img1, img2):
#     width, height = img1.shape[1], img1.shape[0]
#     img2 = cv2.resize(img2, (width, height))
#     num_bins = 10
#
#     hist1 = cv2.calcHist([img1], [0], None, [num_bins], [0, 255])
#     hist2 = cv2.calcHist([img2], [0], None, [num_bins], [0, 255])
#     sum = 0
#     for i in range(num_bins):
#         sum += abs(hist1[i] - hist2[i])
#     return sum / float(width * height)
#
#
# # 3 hist
# def histogram(choice):
#     src_input = cv2.imread("image.orig/examples/" + choice)
#     min_diff = 1e50
#
#     cv2.imshow("Input", src_input)
#
#     # change the image to gray scale
#     src_gray = cv2.cvtColor(src_input, cv2.COLOR_BGR2GRAY)
#
#     # read image database
#     database = sorted(glob(database_dir + "/*.jpg"))
#
#     for img in database:
#         # read image
#         img_rgb = cv2.imread(img)
#         # convert to gray scale
#         img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
#         # compare the two images
#         # diff = compareImgs(src_gray, img_gray)
#         # compare the two images by histogram, uncomment the following line to use histogram
#         diff = compareImgs_hist(src_gray, img_gray)
#         print(img, diff)
#         # find the minimum difference
#         if diff <= min_diff:
#             # update the minimum difference
#             min_diff = diff
#             # update the most similar image
#             closest_img = img_rgb
#             result = img
#
#     print("the most similar image is %s, the pixel-by-pixel difference is %f " % (result, min_diff))
#     print("\n")
#
#     cv2.imshow("Result", closest_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
# # 1, 4, 6 absdiff
# def subtraction(choice):
#     src_input = cv2.imread("image.orig/examples/" + choice)
#     min_diff = 1e50
#
#     cv2.imshow("Input", src_input)
#
#     # change the image to gray scale
#     src_gray = cv2.cvtColor(src_input, cv2.COLOR_BGR2GRAY) + 1.0
#
#     # read image database
#     database = sorted(glob(database_dir + "/*.jpg"))
#
#     for img in database:
#         # read image
#         img_rgb = cv2.imread(img)
#         # convert to gray scale
#         img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY) + 1.0
#         # compare the two images
#         score = mse(src_gray, img_gray)
#         # compare the two images by histogram, uncomment the following line to use histogram
#         # diff = compareImgs_hist(src_gray, img_gray)
#         print(img, score)
#         # if mean > 0.6:
#         #     cv2.imwrite(out_dir+img[11:], img_rgb)
#         # find the minimum difference
#         if score <= min_diff:
#             # update the minimum difference
#             min_diff = score
#             # update the most similar image
#             closest_img = img_rgb
#             result = img
#
#     print("the most similar image is %s, the pixel-by-pixel difference is %f " % (result, min_diff))
#     print("\n")
#
#     cv2.imshow("Result", closest_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
# # SIFT 5 - 0.75
# def sift(choice):
#     src_input = cv2.imread("image.orig/examples/" + choice)
#     max_match = 0
#
#     cv2.imshow("Input", src_input)
#     src_gray = cv2.cvtColor(src_input, cv2.COLOR_BGR2GRAY)
#
#     detector = cv2.SIFT_create()
#     keypoints1, descriptors1 = detector.detectAndCompute(src_input, None)
#     # descriptors1 /= (descriptors1.sum(axis=1, keepdims=True) + 1e-7)
#     # descriptors1 = np.sqrt(descriptors1)
#     # read image database
#     database = sorted(glob(database_dir + "/*.jpg"))
#     bf = cv2.BFMatcher()
#
#     for img in database:
#         # read image
#         img_rgb = cv2.imread(img)
#         # img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
#         # with open("descriptors/" + img[11:-3] + "pkl", 'rb') as f:
#         #     descriptors2 = pickle.load(f)
#         # with open("descriptors_root/"+img[11:-3]+"pkl", 'rb') as f:
#         #     descriptors2 = pickle.load(f)
#         # compare the two images
#         keypoints2, descriptors2 = detector.detectAndCompute(img_rgb, None)
#         # -- Step 2: Matching descriptor vectors with a brute force matcher
#         matches = bf.knnMatch(descriptors1, descriptors2, k=2)
#         # matches = sorted(matches, key=lambda x: x.distance)
#         good_matches = 0
#         for m, n in matches:
#             if m.distance < 0.75 * n.distance:
#                 good_matches += 1
#         similarity = good_matches / len(matches)
#         print(img, similarity)
#         if similarity > 0.03:
#             cv2.imwrite(out_dir+img[11:], img_rgb)
#         # find the minimum difference
#         if similarity > max_match:
#             # update the minimum difference
#             max_match = similarity
#             # update the most similar image
#             result = img
#             closest_img = img_rgb
#
#     print("the most similar image is %s, the pixel-by-pixel difference  " % result)
#     print("\n")
#
#     cv2.imshow("Result", closest_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
# def orb(choice):
#     src_input = cv2.imread("image.orig/examples/" + choice)
#     max_match = 0
#
#     cv2.imshow("Input", src_input)
#     src_gray = cv2.cvtColor(src_input, cv2.COLOR_BGR2GRAY)
#
#     detector = cv2.ORB_create()
#     keypoints1, descriptors1 = detector.detectAndCompute(src_gray, None)
#     # read image database
#     database = sorted(glob(database_dir + "/*.jpg"))
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#
#     for img in database:
#         # read image
#         img_rgb = cv2.imread(img)
#         img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
#         # with open("descriptors/" + img[11:-3] + "pkl", 'rb') as f:
#         #     descriptors2 = pickle.load(f)
#         # compare the two images
#         keypoints2, descriptors2 = detector.detectAndCompute(img_gray, None)
#         # -- Step 2: Matching descriptor vectors with a brute force matcher
#         matches = bf.match(descriptors1, descriptors2)
#         matches = sorted(matches, key=lambda x: x.distance)
#         min_dist = matches[0].distance
#         good_matches = len(tuple(filter(lambda x: x.distance <= 2 * min_dist, matches)))
#         # matches = sorted(matches, key=lambda x: x.distance)
#         # good_matches = 0
#         # for m, n in matches:
#         #     if m.distance < 0.65 * n.distance:
#         #         good_matches += 1
#         similarity = good_matches / len(matches)
#         print(img, similarity)
#         # if similarity > 0.03:
#         #     cv2.imwrite(out_dir+img[11:], img_rgb)
#         # find the minimum difference
#         if similarity > max_match:
#             # update the minimum difference
#             max_match = similarity
#             # update the most similar image
#             result = img
#             closest_img = img_rgb
#
#     print("the most similar image is %s, the pixel-by-pixel difference  " % result)
#     print("\n")
#
#     cv2.imshow("Result", closest_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
# def flann(choice):
#     src_input = cv2.imread("image.orig/examples/" + choice)
#     max_match = 0
#
#     cv2.imshow("Input", src_input)
#     src_gray = cv2.cvtColor(src_input, cv2.COLOR_BGR2GRAY)
#
#     detector = cv2.SIFT_create()
#     keypoints1, descriptors1 = detector.detectAndCompute(src_gray, None)
#     # descriptors1 /= (descriptors1.sum(axis=1, keepdims=True) + 1e-7)
#     # descriptors1 = np.sqrt(descriptors1)
#     # read image database
#     database = sorted(glob(database_dir + "/*.jpg"))
#
#     #FLANN parameters
#     FLANN_INDEX_KDTREE = 0
#     index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
#     search_params = dict(checks=100)  # or pass empty dictionary
#
#     flann = cv2.FlannBasedMatcher(index_params, search_params)
#
#     for img in database:
#         # read image
#         img_rgb = cv2.imread(img)
#         img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
#         with open("descriptors/" + img[11:-3] + "pkl", 'rb') as f:
#             descriptors2 = pickle.load(f)
#         # with open("descriptors_root/"+img[11:-3]+"pkl", 'rb') as f:
#         #     descriptors2 = pickle.load(f)
#         # compare the two images
#         # keypoints2, descriptors2 = detector.detectAndCompute(img_gray, None)
#         # -- Step 2: Matching descriptor vectors with a brute force matcher
#         matches = flann.knnMatch(descriptors1, descriptors2, k=2)
#         # matches = sorted(matches, key=lambda x: x.distance)
#         good_matches = 0
#         for m, n in matches:
#             if m.distance < 0.7 * n.distance:
#                 good_matches += 1
#         similarity = good_matches / len(matches)
#         print(img, similarity)
#         if similarity > 0.03:
#             cv2.imwrite(out_dir+img[11:], img_rgb)
#         # find the minimum difference
#         if similarity > max_match:
#             # update the minimum difference
#             max_match = similarity
#             # update the most similar image
#             result = img
#             closest_img = img_rgb
#
#     print("the most similar image is %s, the pixel-by-pixel difference  " % result)
#     print("\n")
#
#     cv2.imshow("Result", closest_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
# def ssim(choice):
#     src_input = cv2.imread("image.orig/examples/" + choice)
#     max_score = 0
#
#     cv2.imshow("Input", src_input)
#
#     # change the image to gray scale
#     src_gray = cv2.cvtColor(src_input, cv2.COLOR_BGR2GRAY)
#
#     # read image database
#     database = sorted(glob(database_dir + "/*.jpg"))
#
#     for img in database:
#         # read image
#         img_rgb = cv2.imread(img)
#         # convert to gray scale
#         img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
#         img_gray = cv2.resize(img_gray, (src_gray.shape[1], src_gray.shape[0]))
#
#         (score, diff) = structural_similarity(src_gray, img_gray, win_size=101, full=True)
#         print(img, score)
#         # find the minimum difference
#         if score > max_score:
#             # update the minimum difference
#             max_score = score
#             # update the most similar image
#             closest_img = img_rgb
#             result = img
#
#     print("the most similar image is %s, the pixel-by-pixel difference is %f " % (result, max_score))
#     print("\n")
#
#     cv2.imshow("Result", closest_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


def dl(choice):
    with open("features/" + choice[:-3] + "pkl", 'rb') as f:
        src = pickle.load(f)
    # src_input = cv2.imread("image.orig/examples/" + choice)
    min_dc = 1e50

    # read image database
    database = sorted(glob(database_dir + "/*.jpg"))
    metric = 'cosine'
    record = []

    for img in database:
        # read image
        # img_rgb = cv2.imread(img)
        with open("features/" + img[19:-3] + "pkl", 'rb') as f:
            dc = distance.cdist([src], [pickle.load(f)], metric)[0]
        record.append([img, dc])
        print(img, dc)

        # find the minimum difference
        if dc < min_dc:
            # update the minimum distance
            min_dc = dc
            # update the most similar image
            # closest_img = img_rgb
            result = img
    return result, record
    # record.sort(key=lambda x: x[1])
    # for i in range(100):
    #     cv2.imwrite(out_dir+record[i][0][19:], cv2.imread(record[i][0]))
    # print("the most similar image is %s, the pixel-by-pixel difference is %f " % (result, min_dc))
    # print("\n")

    # cv2.imshow("Result", closest_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def retrieve(choice):
    # print("1: Image retrieval demo")

    # # to simplify if clauses
    # dic = {1: "beach.jpg", 2: "building.jpg", 3: "bus.jpg", 4: "dinosaur.jpg",
    #        5: "flower.jpg", 6: "horse.jpg", 7: "man.jpg"}

    # print("1: beach")
    # print("2: building")
    # print("3: bus")
    # print("4: dinosaur")
    # print("5: flower")
    # print("6: horse")
    # print("7: man")
    # choice = eval(input("Type in the number to choose a category and type enter to confirm\n"))

    # print("You choose: %s\n" % dic[choice])

    dl(choice)
    # ssim(dic[choice])

# if __name__ == '__main__':
#     retrieve()

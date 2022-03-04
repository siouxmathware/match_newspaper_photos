import numpy as np
import cv2
from preprocess import ImageSignature


def confirm_found_match(image1, image2):
    """
    input: two grayscale images
    find matching SIFT features into images
    find homography and checks if rectangle transforms to rectangle
    params hard-coded: minimum number of good features, 
    maximum scalar product to consider 90 deg
    return: True/False of the transform
    """
    match_sift = False
    n_good = 12
    min_dot = 0.3

    obj = ImageSignature(image1)
    img1, mask1 = obj.sift_image_mask()
    h, w = img1.shape
    
    obj = ImageSignature(image2)
    img2, mask2 = obj.sift_image_mask()
    
    sift = cv2.SIFT_create()  # cv2.xfeatures2d.SIFT_create()  # older version
    # find the keypoints and descriptors with SIFT; mask to discard the borders
    kp1, des1 = sift.detectAndCompute(img1, mask=mask1)
    kp2, des2 = sift.detectAndCompute(img2, mask=mask2)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matches_mask = [[0, 0] for _ in matches]

    # ratio test as per Lowe's paper
    good = []
    pts1 = []
    pts2 = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matches_mask[i] = [1, 0]
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    if len(good) > n_good:
        src_pnt = np.array(pts1).reshape(-1, 1, 2)
        dst_pnt = np.array(pts2).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pnt, dst_pnt, cv2.RANSAC, 5.0)
        # check that rectangle transforms to rectangle: angles preserved
        if M is not None:
            src_corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst_corners = cv2.perspectiveTransform(src_corners, M)
            xa, xb, xc, xd = np.array([[x, y] for (x, y) in dst_corners.reshape(4, 2)])
            vec_a = xb - xa
            vec_b = xc - xb
            vec_c = xd - xc
            vec_d = xa - xd
            # KMYE: What do each of the four lines below compute?
            cos_ab = -np.dot(vec_a, vec_b)/np.sqrt(np.dot(vec_a, vec_a)*np.dot(vec_b, vec_b))
            cos_cb = -np.dot(vec_c, vec_b)/np.sqrt(np.dot(vec_c, vec_c)*np.dot(vec_b, vec_b))
            cos_cd = -np.dot(vec_c, vec_d)/np.sqrt(np.dot(vec_c, vec_c)*np.dot(vec_d, vec_d))
            cos_ad = -np.dot(vec_a, vec_d)/np.sqrt(np.dot(vec_a, vec_a)*np.dot(vec_d, vec_d))
            match_sift = (abs(cos_ab) < min_dot) and (abs(cos_cb) < min_dot) and (abs(cos_cd) < min_dot) and (abs(cos_ad) < min_dot)
    return match_sift

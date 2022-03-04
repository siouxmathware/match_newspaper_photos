import numpy as np
import cv2

from tensorflow.keras.applications.vgg16 import preprocess_input


class ImageSignature:
    """
    Set of functions to preprocess image (3-channel)
    extract hashes and features
    TODO different for foto/krant
    """
    def __init__(self, image, im_size=(224, 224), sift_imsize=512):
        self.image = image.copy()
        self.im_size = im_size
        self.sift_imsize = sift_imsize

    def histogram_equalization(self):
        img_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(img_gray)
    
    def normalize_image(self):
        img = cv2.GaussianBlur(self.image, (7, 7), 0)
        norm_img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return (255*norm_img).astype(np.uint8)

    def cnn_image(self):
        img = self.normalize_image()
        img_resize = cv2.resize(img, self.im_size)
        img_expand = np.expand_dims(np.float32(img_resize), axis=0)
        return preprocess_input(img_expand)
    
    def sift_image_mask(self):
        norm_img = self.normalize_image()
        h, w = norm_img.shape
        img_resize = cv2.resize(norm_img, (int(self.sift_imsize*w/h), self.sift_imsize), interpolation=cv2.INTER_AREA)
        mask = 0 * img_resize
        h, w = mask.shape
        delta = min([h, w])//10
        mask[delta:-delta, delta:-delta] = 255
        return img_resize, mask

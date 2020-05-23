#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN - Inspect Balloon Training Data
# 
# Inspect and visualize data loading and pre-processing code.

# In[1]:

import os
import sys
import random
import numpy as np
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = True
# Root directory of the project
ROOT_DIR = os.path.abspath("../")
image_folder = "mask_results/"
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from src.mrcnn import utils
from src.mrcnn import visualize
import os
from src.mask.samples.balloon import balloon
import cv2


def crop_image(img, tol=0):
    # img is 2D image data
    # tol  is tolerance
    nmask = img > tol
    return img[np.ix_(nmask.any(1), nmask.any(0))]


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def begin():
    images = []
    config = balloon.BalloonConfig()
    BALLOON_DIR = os.path.join(ROOT_DIR, "dataset")

    dataset = balloon.BalloonDataset()
    dataset.load_balloon(BALLOON_DIR, "train")

    dataset.prepare()

    print("Image Count: {}".format(len(dataset.image_ids)))
    print("Class Count: {}".format(dataset.num_classes))
    for i, info in enumerate(dataset.class_info):
        print("{:3}. {:50}".format(i, info['name']))

    image_id = random.choice(dataset.image_ids)
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    # Compute Bounding box
    bbox = utils.extract_bboxes(mask)

    # Display image and add`itional stats
    # print("image_id ", image_id, dataset.image_reference(image_id))
    # log("image", image)
    # log("mask", mask)
    # log("class_ids", class_ids)
    # log("bbox", bbox)
    image_ids = np.random.choice(dataset.image_ids, 4)
    for image_id in image_ids:
        image = dataset.load_image(image_id)
        mask, class_ids = dataset.load_mask(image_id)
        visualize.mask_gen(image, mask, class_ids, dataset.class_names)
        newimg = mask * 255
        newimg = np.array(newimg, dtype='uint8')
        np.set_printoptions(threshold=sys.maxsize)
        cv2.imshow('s', newimg)
        cv2.waitKey()
        try:
            cv2.imwrite('buffer.png', newimg)
            readimg = cv2.imread('buffer.png', 0)
            crop = crop_image(readimg)
            os.remove('buffer.png')
        except:
            pass
        cv2.imshow('a', crop)
        images.append(crop)
    print(images)
    return images
    # cv2.imwrite("{}{}cat-{}.png".format(ROOT_DIR,image_folder,time.time()).replace(".","-",1), crop)
    # ^^^^^ FORMAT FOR SAVING
    # cv2.imwrite('img.png', newimg)
    # img = cv2.imread('img.png', 0)
    # print(img.shape)
    # crop = crop_image(img)
    # print()
    # cv2.imshow('s', crop)
    # cv2.waitKey()
    # print("prev: {} {}, new: {} {} ".format(len(newimg), len(newimg[0]), len(crop), len(crop[0])))
    # Display image and instances
    # visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)


begin()

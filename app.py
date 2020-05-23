import os
import cv2
import glob
import time
from matplotlib import pyplot
from src.util.Image_File import Image_File
import matplotlib.pyplot as plt
import numpy as np
import argparse
from src.profile_classifier.svm import fit_svm, predict
from scipy.stats import linregress, pearsonr
from PIL import Image


def mask(dataset):
    from src.mask.samples.cat import cat
    images = []
    for image in glob.glob('{}*.jpg'.format(dataset)):
        images.append(Image_File(image, cv2.imread(image)))

    model = cat.load_model("coco", "samples/cat/dataset")
    images = cat.run(model, images)

    # Removing background pixels from mask
    for image in images:
        if image.mask_full is not None:
            x_point = []
            y_point = []
            for y, row in enumerate(image.mask_full):
                for x, col in enumerate(row):
                    if col != 0:
                        x_point.append(x)
                        y_point.append(y)

            image.mask_crop = [x_point, y_point]

    return images


def linear_regression(images):
    import src.description.description as lr
    return lr.run(images)


def predict_profile(state, image_path):
    print("path: {}".format(image_path))
    # images
    if state == 0:
        predict_mask(image_path)

    # single image
    if state == 1:
        xmask = single_pred_mask(image_path)
        result = linear_regression(xmask)
        tx = []
        ty = []
        if result.curve is not None:
            for x, y in result.curve:
                tx.append(x)
                ty.append(y)
            pr = pearsonr(tx, ty)
            print(result.coef)
            print("Predicted class: {}".format(predict(pr)))


def generate_masks(state, image_path):
    print("path: {}".format(image_path))
    # images
    if state == 0:
        m = mask(image_path)
        for i in m:
            if i is not None:
                if i.mask_full is not None:
                    im_rgb = cv2.cvtColor(np.uint8(i.mask_full), cv2.COLOR_BGR2RGB)
                    Image.fromarray(im_rgb).save("output/mask-{}.jpg".format(i.name.split("\\")[1].split(".")[0]))



    # single image
    if state == 1:
        xmask = single_pred_mask(image_path)
        try:
            im_rgb = cv2.cvtColor(np.uint8(xmask.mask_full), cv2.COLOR_BGR2RGB)
            Image.fromarray(im_rgb).save("output/mask-{}.jpg".format(xmask.name.split("\\")[1].split(".")[0]))
        except:
            print("No mask for {}".format(xmask.name))


def generate(dataset="input/"):
    print("Generating dataset...")
    images = mask(dataset)
    images = linear_regression(images)
    coefs = []
    nt = []
    prs = []
    for image in images:
        if image is not None:
            tx = []
            ty = []
            for coef in image.coef:
                coefs.append(coef)
            for x, y in image.curve:
                tx.append(x)
                ty.append(y)
            pr = pearsonr(tx, ty)
            prs.append(pr)
            print(pr)

            if abs(pr[0] - pr[1]) < 8.5e-1:
                nt.append(0)
            else:
                nt.append(1)
    print(nt)
    fit_svm(prs, nt)

    return images


def predict_mask(images):
    images = mask(images)
    images = linear_regression(images)
    coefs = []
    nt = []
    prs = []
    for image in images:
        if image is not None:
            tx = []
            ty = []
            for coef in image.coef:
                coefs.append(coef)
            for x, y in image.curve:
                tx.append(x)
                ty.append(y)
            pr = pearsonr(tx, ty)
            prs.append(pr)
            image.pr = pr
            image.predict = predict(pr)

    for image in images:
        try:
            if image.predict is not None:
                print(image.name, image.predict)
        except:
            pass

    return images


def single_pred_mask(image):
    from src.mask.samples.cat import cat
    model = cat.load_model("coco", "samples/cat/dataset")
    im = Image_File(image, cv2.imread(image))
    result = cat.run(model, [im])
    result = result[0]

    if result.mask_full is not None:
        x_point = []
        y_point = []
        for y, row in enumerate(result.mask_full):
            for x, col in enumerate(row):
                if col != 0:
                    x_point.append(x)
                    y_point.append(y)

        result.mask_crop = [x_point, y_point]

    return result


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Predict an ideal profile or generate a dataset of profiles.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'predict' or 'generate' or 'mask'")
    parser.add_argument('--image', required=False,
                        metavar="path to image",
                        help="single image path")
    parser.add_argument('--images', required=False,
                        metavar="path to images",
                        help="path to image directory")
    parser.add_argument('--dataset', required=False,
                        metavar="dataset",
                        help='dataset for classification generation')
    args = parser.parse_args()
    if args.command == "generate":
        if args.dataset is None:
            print("Using default input path")
            generate()
        else:
            generate(args.dataset)

    if args.command == "predict":
        assert args.image or args.images, "Argument --image or --images is required for prediction"
        if args.image:
            predict_profile(1, args.image)
        elif args.images:
            assert os.path.isdir(args.images)
            predict_profile(0, args.images)

    if args.command == "mask":
        assert args.image or args.images, "Argument --image or --images is required for mask generation"
        if args.image:
            generate_masks(1, args.image)
        elif args.images:
            assert os.path.isdir(args.images)
            generate_masks(0, args.images)

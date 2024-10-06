import numpy as np
import pandas as pd
from pandas import cut

import cv2
from PIL import Image

import torch
from torchvision.transforms import v2

import datetime
import pickle

from uwimg import *
from sportsanalyzer_utils import max_frame_count


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
object_detection_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom]


FRAME_PATH = 'data/frames/%d.jpg'
# matrices_file = 'relative_matrices_file'

# # # best so far: 3185
# play_number = 0
# view = 0
#
# min_frame = 571
# max_frame = 1171


def image_ops(image):
    # curr_im = convolve_image(curr_im, make_sharpen_filter(), 1)
    # clamp_image(curr_im)

    cutoff(image, 0.666)
    return image


def film_panorama(verbose=False):
    im = load_image(FRAME_PATH % min_frame)

    H = identity_homography()
    curr_H = H

    if verbose:
        matrix_print(H)

    curr_im = load_image(FRAME_PATH % min_frame)
    curr_im = image_ops(curr_im)

    for frame in range(min_frame + 1, max_frame+1):
        start_time = datetime.datetime.now()

        next_im = load_image(FRAME_PATH % frame)
        next_im = image_ops(next_im)

        # im1copy = copy_image(curr_im)
        # im2copy = copy_image(next_im)

        # m = find_and_draw_matches(im1copy, im2copy, 2, 50, 3)
        # save_image(m, "data/matches/%d_%d" % (frame, (frame+1)))

        save_image(next_im, 'tst/%d_target' % frame)

        h_new = relative_homography(curr_im, next_im, thresh=2, iters=75000, inlier_thresh=3, cutoff=50)
        H_new = matrix_mult(curr_H, h_new)

        if verbose:
            print('x')
            matrix_print(h_new)

        im1_transformed = project_image(curr_im, invert_matrix(h_new))

        h_adjust = relative_homography(im1_transformed, next_im, thresh=2, iters=50000, inlier_thresh=3, cutoff=100)
        H_new = matrix_mult(H_new, h_adjust)

        im1_final = project_image(im1_transformed, invert_matrix(h_adjust))
        save_image(im1_final, 'tst/%d_transformed' % frame)

        if verbose:
            print('x')
            matrix_print(h_adjust)

            print('=')
            matrix_print(H_new)

        curr_H = H_new

        curr_im = next_im

        homography_time = datetime.datetime.now()

        if verbose:
            print('homography calculation time:', homography_time - start_time)

        img = FRAME_PATH % frame  # or file, Path, PIL, OpenCV, numpy, list
        # Inference
        results = object_detection_model(img)

        # Results
        objects = results.pred[0]  # or .show(), .save(), .crop(), .pandas(), etc.

        img = make_image(1920, 1080, 3)

        for k in range(3):
            for j in range(1080):
                for i in range(1920):
                    set_pixel(img, i, j, k, 0)

        # print(objects)

        for index in range(objects.shape[0]):
            detected_object = objects[index]

            # print(detected_object)

            if detected_object[5] == 0:
                x = int((detected_object[0] + detected_object[2]) / 2)
                y = int((detected_object[1] + detected_object[3]) / 2)

                color = (frame / max_frame) * 5

                for i in range(-3, 3):
                    for j in range(-3, 3):
                        if color < 1:
                            set_pixel(img, x + i, y + j, 0, 1)
                            set_pixel(img, x + i, y + j, 1, 0)
                            set_pixel(img, x + i, y + j, 2, color)
                        elif color < 2:
                            set_pixel(img, x + i, y + j, 0, 2-color)
                            set_pixel(img, x + i, y + j, 1, 0)
                            set_pixel(img, x + i, y + j, 2, 1)
                        elif color < 3:
                            set_pixel(img, x + i, y + j, 0, 0)
                            set_pixel(img, x + i, y + j, 1, color-2)
                            set_pixel(img, x + i, y + j, 2, 1)
                        elif color < 4:
                            set_pixel(img, x + i, y + j, 0, 0)
                            set_pixel(img, x + i, y + j, 1, 1)
                            set_pixel(img, x + i, y + j, 2, 4-color)
                        else:
                            set_pixel(img, x + i, y + j, 0, color-4)
                            set_pixel(img, x + i, y + j, 1, 1)
                            set_pixel(img, x + i, y + j, 2, 0)

        object_detection_time = datetime.datetime.now()

        if verbose:
            print('object detection time:', object_detection_time - homography_time)

        im = combine_images(im, img, curr_H)
        save_image(im, 'data/movements/%d' % frame)

        if verbose:
            print('movement calculation time:', datetime.datetime.now() - object_detection_time)
            print(frame, 'completed, total time:', datetime.datetime.now() - start_time)

    return curr_H


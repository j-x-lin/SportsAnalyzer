import numpy as np

import cv2
from PIL import Image

import torch
from ultralytics import YOLO

import datetime

from uwimg import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using', device)

object_detection_model = YOLO('yolo11s.pt')


FRAME_PATH = 'data/frames/%d.jpg'


def image_ops(image):
    # image = convolve_image(image, make_sharpen_filter(), 1)
    # clamp_image(image)

    cutoff(image, 0.666)
    return image


# scale a homography by a factor of s
def scale_homography(homography, s):
    # computation for original size homography = S * H * S^-1
    # for scale factor s the scaling matrix S is [[s, 0, 0], [0, s, 0], [0, 0, 1]]
    homography.data[0][2] *= s
    homography.data[1][2] *= s
    homography.data[2][0] /= s
    homography.data[2][1] /= s


# compute homography between curr_im and next_im
# s: scale factor to scale both images down (for efficiency, but at a slightly reduced accuracy)
def compute_homography(curr_im, next_im, s=1):
    if s == 1:
        return relative_homography(curr_im, next_im, sigma=2, thresh=2, nms=3, inlier_thresh=3, iters=50000, cutoff=100)
    else:
        small_curr = bilinear_resize(curr_im, int(curr_im.w / s), int(curr_im.h / s))
        small_next = bilinear_resize(next_im, int(next_im.w / s), int(next_im.h / s))

        h_new = relative_homography(small_curr, small_next, sigma=2, thresh=2, nms=3, inlier_thresh=3, iters=50000,
                                    cutoff=100)

        # scale up homography by s
        scale_homography(h_new, s)

        free_image(small_curr)
        free_image(small_next)

        return h_new


# min_frame, max_frame: the frames to start at and end at (INCLUSIVE)
def film_panorama(min_frame, max_frame, verbose=False, save_debug_images=False):
    im = load_image(FRAME_PATH % min_frame)

    H = identity_homography()

    if verbose:
        matrix_print(H)

    curr_im = load_image(FRAME_PATH % min_frame)
    curr_im = image_ops(curr_im)

    for frame in range(min_frame + 1, max_frame+1):
        start_time = datetime.datetime.now()

        next_im = load_image(FRAME_PATH % frame)
        next_im = image_ops(next_im)

        if save_debug_images:
            im1copy = copy_image(curr_im)
            im2copy = copy_image(next_im)

            m = find_and_draw_matches(im1copy, im2copy, 2, 50, 3)
            save_image(m, "data/matches/%d_%d" % (frame, (frame+1)))

            save_image(next_im, 'tst/%d_target' % frame)

            free_image(im1copy)
            free_image(im2copy)
            free_image(m)

        h_new = compute_homography(curr_im, next_im, s=4)

        if verbose:
            print('x')
            matrix_print(h_new)

        H_new = matrix_mult(H, h_new)

        im1_transformed = project_image(curr_im, invert_matrix(h_new))
        h_adjust = compute_homography(im1_transformed, next_im, s=1)

        H = matrix_mult(H_new, h_adjust)

        if save_debug_images:
            im1_final = project_image(im1_transformed, invert_matrix(h_adjust))
            save_image(im1_final, 'tst/%d_transformed' % frame)

        if verbose:
            print('x')
            matrix_print(h_adjust)

            print('=')
            matrix_print(H)

        free_image(im1_transformed)
        free_image(curr_im)

        free_matrix(h_new)
        free_matrix(h_adjust)

        curr_im = next_im

        homography_time = datetime.datetime.now()

        if verbose:
            print('homography calculation time:', homography_time - start_time)

        obj_dots = make_image(1920, 1080, 3)

        for k in range(3):
            for j in range(1080):
                for i in range(1920):
                    set_pixel(obj_dots, i, j, k, 0)

        # Inference
        img = FRAME_PATH % frame  # or file, Path, PIL, OpenCV, numpy, list
        results = object_detection_model.predict(source=img, stream=True)

        # Results
        for result in results:
            # print(detected_object)
            bounding_boxes = result.boxes.xyxy

            for i in range(len(bounding_boxes)):
                if result.boxes.cls[i] == 0:
                    x = int((bounding_boxes[i][0] + bounding_boxes[i][2]) / 2)
                    y = int((bounding_boxes[i][1] + bounding_boxes[i][3]) / 2)

                    color = ((frame-min_frame) / (max_frame-min_frame)) * 5

                    for i in range(-3, 3):
                        for j in range(-3, 3):
                            if color < 1:
                                set_pixel(obj_dots, x + i, y + j, 0, 1)
                                set_pixel(obj_dots, x + i, y + j, 1, 0)
                                set_pixel(obj_dots, x + i, y + j, 2, color)
                            elif color < 2:
                                set_pixel(obj_dots, x + i, y + j, 0, 2-color)
                                set_pixel(obj_dots, x + i, y + j, 1, 0)
                                set_pixel(obj_dots, x + i, y + j, 2, 1)
                            elif color < 3:
                                set_pixel(obj_dots, x + i, y + j, 0, 0)
                                set_pixel(obj_dots, x + i, y + j, 1, color-2)
                                set_pixel(obj_dots, x + i, y + j, 2, 1)
                            elif color < 4:
                                set_pixel(obj_dots, x + i, y + j, 0, 0)
                                set_pixel(obj_dots, x + i, y + j, 1, 1)
                                set_pixel(obj_dots, x + i, y + j, 2, 4-color)
                            else:
                                set_pixel(obj_dots, x + i, y + j, 0, color-4)
                                set_pixel(obj_dots, x + i, y + j, 1, 1)
                                set_pixel(obj_dots, x + i, y + j, 2, 0)

        object_detection_time = datetime.datetime.now()

        if verbose:
            print('object detection time:', object_detection_time - homography_time)

        im = combine_images(im, obj_dots, H)

        if save_debug_images:
            save_image(im, 'tst/%d_movement' % frame)

        if verbose:
            print('movement calculation time:', datetime.datetime.now() - object_detection_time)
            print(frame, 'completed, total time:', datetime.datetime.now() - start_time)

        free_image(obj_dots)

    return im

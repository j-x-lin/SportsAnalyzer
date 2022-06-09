from pandas import cut
from uwimg import *
import numpy as np
import torch
import pickle

import pandas as pd

print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))


import torch
import pickle
from uwimg import *

max_frame = 30

def film_panorama():
    with open('relative_matrices_file', 'wb') as relative_matrices_file:
        matrices = []
        matrices.append(matrix_to_array(identity_homography()))
        matrix_print(array_to_matrix(matrices[0]))

        for i in range(1, max_frame):
            im1 = load_image("data/1/" + str(i) + ".jpg")
            im1 = convolve_image(im1, make_sharpen_filter(), 1)
            clamp_image(im1)

            im2 = load_image("data/1/" + str(i+1) + ".jpg")
            im2 = convolve_image(im2, make_sharpen_filter(), 1)
            clamp_image(im2)

            im1copy = copy_image(im1)
            im2copy = copy_image(im2)

            m = find_and_draw_matches(im1, im2, 2, 50, 3)
            save_image(m, "matches" + str(i) + "_" + str(i+1))

            H = array_to_matrix(matrices[i-1])
            h_new = relative_homography(im1, im2, thresh=2, iters=50000, inlier_thresh=2, cutoff=100)

            matrices.append(matrix_to_array(matrix_mult(H, h_new)))
            matrix_print(array_to_matrix(matrices[i]))
            print(i, 'completed')

        pickle.dump(matrices, relative_matrices_file)

    return matrices

# matrices = film_panorama()

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom]

im = load_image("data/1/1.jpg")

with open('relative_matrices_file', 'rb') as relative_matrices_file:
    # Step 3
    matrices_loaded = pickle.load(relative_matrices_file)
    print(len(matrices_loaded))

    # Images
    for frame in range(1, max_frame+1):
        print(frame)

        img = 'data/1/' + str(frame) + '.jpg'  # or file, Path, PIL, OpenCV, numpy, list

        # Inference
        results = model(img)

        # Results
        objects = results.pred[0]  # or .show(), .save(), .crop(), .pandas(), etc.

        img = make_image(1280, 720, 3)

        for k in range(3):
            for j in range(720):
                for i in range(1280):
                    set_pixel(img, i, j, k, 1)
        
        # print(objects)

        for index in range(objects.shape[0]):
            object = objects[index]

            # print(object)

            if object[5] == 0:
                x = int((object[0] + object[2]) / 2)
                y = int((object[1] + object[3]) / 2)

                for i in range(-3, 3):
                    for j in range(-3, 3):
                        color = 3 - (float(frame) / float(max_frame) * 3)

                        if color > 2:
                            set_pixel(img, x+i, y+j, 0, color - 2)
                            set_pixel(img, x+i, y+j, 1, 1)
                            set_pixel(img, x+i, y+j, 2, 1)
                        elif color > 1:
                            set_pixel(img, x+i, y+j, 0, 0)
                            set_pixel(img, x+i, y+j, 1, color-1)
                            set_pixel(img, x+i, y+j, 2, 1)
                        else:
                            set_pixel(img, x+i, y+j, 0, 0)
                            set_pixel(img, x+i, y+j, 1, 0)
                            set_pixel(img, x+i, y+j, 2, color)

                        
        
        im = combine_images(im, img, array_to_matrix(matrices_loaded[frame-1]))

        save_image(im, 'movements')

import torch
import pickle
from uwimg import *

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom]

with open('relative_matrices_file', 'rb') as relative_matrices_file:
    # Step 3
    matrices_loaded = pickle.load(relative_matrices_file)

    # Images
    for im in range(1, 217):
        img = 'data/1/' + str(im) + '.jpg'  # or file, Path, PIL, OpenCV, numpy, list

        # Inference
        results = model(img)

        # Results
        objects = results.pred  # or .show(), .save(), .crop(), .pandas(), etc.

        img = make_image(1280, 720, 3)

        for k in range(3):
            for j in range(720):
                for i in range(1280):
                    set_pixel(img, 1, 1, 1)
        
        print(objects.__dict__)
        break

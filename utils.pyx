import os.path

import numpy as np

from PIL import Image

import torch
from torchvision.transforms import v2


def max_frame_count():
    max_frame = 1
    min_frame = 0

    while os.path.isfile("data/frames/%d.jpg" % max_frame):
        min_frame = max_frame
        max_frame *= 2

    while min_frame < max_frame - 1:
        mid = int((max_frame + min_frame) / 2)
        if os.path.isfile("data/frames/%d.jpg" % mid):
            min_frame = mid
        else:
            max_frame = mid

    return min_frame


# current mean: [0.49867509 0.59165666 0.37727006]
# current STD: [0.01694534 0.03099744 0.01660505]
def calculate_mean_std():
    data_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32)
    ])

    means = np.zeros(3)
    nVar = np.zeros(3)
    num_images = max_frame_count() + 1
    for i in range(num_images):
        image = data_transforms(Image.open('data/frames/%d.jpg' % i)).numpy()

        image_mean = np.mean(image, axis=(1, 2))

        means += image_mean
        nVar += image_mean ** 2

        print(means / (i+1))
        print(np.sqrt(nVar / (i+1) - (means / (i+1)) ** 2))

    means /= num_images

    return means, np.sqrt(nVar / num_images - means ** 2)

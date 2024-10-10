import numpy as np

import torch

from PIL import Image

from framesplitter import split_frames
from pipeline import film_panorama
from viewrecognizer import get_view_recognizer_model, get_view_recognizer_data_transforms

from uwimg import *
from sportsanalyzer_utils import max_frame_count

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using', device)

view_recognizer_data_transforms = get_view_recognizer_data_transforms()['val']
view_recognizer_model = get_view_recognizer_model()

# TODO: uncomment later
# split_frames()

play_number = 1
view = 0
start_frame = 0

for frame in range(max_frame_count() + 1):
    image = Image.open('data/frames/%d.jpg' % frame)
    image = view_recognizer_data_transforms(image)

    data = torch.tensor(np.array([image])).to(device)
    result = torch.argmax(view_recognizer_model(data)).item()

    if not result == view:
        end_frame = frame-1
        im = film_panorama(start_frame, end_frame, False, False)
        save_image(im, 'data/movements/%d' % play_number)
        free_image(im)

        print(play_number, 'from', start_frame, 'to', end_frame, 'DONE')

        view = result
        play_number += 1
        start_frame = frame

# best so far: 3185
print('Total plays detected:', play_number)

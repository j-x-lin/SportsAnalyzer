import numpy as np
import concurrent.futures

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


def analyze_play(start_frame, end_frame, play_number):
    movements = film_panorama(start_frame, end_frame, False, False)
    save_image(movements, 'data/movements/%d' % play_number)
    free_image(movements)

    print(play_number, 'from', start_frame, 'to', end_frame, 'DONE')


play_number = 1
view = 0
start_frame = 0

thread_list = []

split_frames()

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as threadpool:
    for frame in range(max_frame_count() + 1):
        image = Image.open('data/frames/%d.jpg' % frame)
        image = view_recognizer_data_transforms(image)

        data = torch.tensor(np.array([image])).to(device)
        result = torch.argmax(view_recognizer_model(data)).item()

        if not result == view:
            threadpool.submit(analyze_play, start_frame, frame-1, play_number)

            view = result
            play_number += 1
            start_frame = frame

    print('Total plays detected:', play_number)

    threadpool.shutdown()

    # best so far: 3185
    print('DONE')

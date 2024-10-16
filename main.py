import numpy as np
import multiprocessing

import torch
from PIL import Image

from framesplitter import split_frames
from pipeline import film_panorama
from viewrecognizer import get_view_recognizer_model, get_view_recognizer_data_transforms

from uwimg import *
from sportsanalyzer_utils import max_frame_count


def analyze_play(start_frame, end_frame, play_number):
    print(play_number, 'from', start_frame, 'to', end_frame, 'starting')
    movements = film_panorama(start_frame, end_frame, False, False)
    save_image(movements, 'data/movements/%d' % play_number)
    free_image(movements)

    print(play_number, 'from', start_frame, 'to', end_frame, 'DONE')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    print('Using', device)

    view_recognizer_data_transforms = get_view_recognizer_data_transforms()
    view_recognizer_model = get_view_recognizer_model()

    play_number = 1
    view = 0
    start_frame = 0
    max_frame = max_frame_count()

    split_frames()

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()-1) as threadpool:
        for frame in range(max_frame + 1):
            image = Image.open('data/frames/%d.jpg' % frame)
            image = view_recognizer_data_transforms(image)

            data = torch.tensor(np.array([image])).to(device)
            result = torch.argmax(view_recognizer_model(data)).item()

            if not result == view:
                threadpool.apply_async(analyze_play, [start_frame, frame-1, play_number])

                view = result
                play_number += 1
                start_frame = frame

        threadpool.apply_async(analyze_play, [start_frame, max_frame, play_number])

        threadpool.close()

        # best so far: 3012
        print('Total plays detected:', play_number)

        threadpool.join()

        print('DONE')

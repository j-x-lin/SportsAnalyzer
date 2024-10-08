from framesplitter import split_frames
from pipeline import film_panorama

# TODO: comment out later
from uwimg import *

# TODO: uncomment later
# split_frames()

final_matrix = film_panorama(True, False)

print('---FINAL MATRIX---')
matrix_print(final_matrix)

print('DONE')

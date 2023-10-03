# #include <stdio.h>
# #include <string.h>
# #include <assert.h>
# #include <math.h>
# #include "image.h"
#
# def get_pixel(im, x, y, c):
#     if x < 0:
#         x = 0
#     elif x >= im.w:
#         x = im.w - 1
#
#     if y < 0:
#         y = 0
#     elif y >= im.h:
#         y = im.h - 1
#
#     return im.data[x + im.w * y + im.w * im.h * c]
#
# def set_pixel(im, x, y, c, v):
#     if x < 0 or x >= im.w:
#         return
#
#     if y < 0 or y >= im.h:
#         return
#
#     im.data[x + im.w * y + im.w * im.h * c] = v
#     return
#
# def copy_image(im):
#     copy = make_image(im.w, im.h, im.c)
#     for i in range(im.w * im.h * im.c):
#         copy.data[i] = im.data[i]
#
#     return copy
#
# def reshape_image(im, w, h):
#     copy = make_image(w, h, im.c)
#
#     for k in range(im.c):
#         for j in range(h):
#             for i in range(w):
#                 set_pixel(copy, i, j, k, get_pixel(im, i, j, k));
#
#     return copy
#
# image rgb_to_grayscale(image im)
# {
#     assert(im.c == 3);
#     image gray = make_image(im.w, im.h, 1);
#
#     int c_size = im.w * im.h;
#
#     // TODO Fill this in
#     for (int i = 0; i < c_size; i++)
#     {
#         gray.data[i] = 0.299 * im.data[i] + 0.587 * im.data[i+c_size] + 0.114 * im.data[i+2*c_size];
#     }
#
#     return gray;
# }
#
# void shift_image(image im, int c, float v)
# {
#     // TODO Fill this in
#     int c_size = im.w * im.h;
#
#     for (int i = c_size * c; i < c_size * c + c_size; i++)
#     {
#         im.data[i] += v;
#     }
# }
#
# void clamp_image(image im)
# {
#     // TODO Fill this in
#     for (int i = 0; i < im.w * im.h * im.c; i++)
#     {
#         if (im.data[i] < 0) {
#             im.data[i] = 0;
#         } else if (im.data[i] > 1) {
#             im.data[i] = 1;
#         }
#     }
# }
#
#
# // These might be handy
# float three_way_max(float a, float b, float c)
# {
#     return (a > b) ? ( (a > c) ? a : c) : ( (b > c) ? b : c) ;
# }
#
# float three_way_min(float a, float b, float c)
# {
#     return (a < b) ? ( (a < c) ? a : c) : ( (b < c) ? b : c) ;
# }
#
# void rgb_to_hsv(image im)
# {
#     // TODO Fill this in
#     int c_size = im.w * im.h;
#
#     for (int i = 0; i < c_size; i++)
#     {
#         float R = im.data[i];
#         float G = im.data[i+c_size];
#         float B = im.data[i+2*c_size];
#
#         float V = three_way_max(R, G, B);
#         im.data[i+2*c_size] = V;
#
#         float m = three_way_min(R, G, B);
#         float C = V-m;
#
#         if (V == 0) {
#             im.data[i+c_size] = 0;
#         } else {
#             im.data[i+c_size] = C/V;
#         }
#
#         float H1;
#
#         if (C == 0) {
#             H1 = 0;
#         } else if (V == R) {
#             H1 = (G-B)/C;
#         } else if (V == G) {
#             H1 = (B-R)/C + 2;
#         } else if (V == B) {
#             H1 = (R-G)/C + 4;
#         } else {
#             H1 = 0;
#         }
#
#         if (H1 < 0) {
#             im.data[i] = H1/6 + 1;
#         } else {
#             im.data[i] = H1/6;
#         }
#     }
# }
#
# void hsv_to_rgb(image im)
# {
#     // TODO Fill this in
#     int c_size = im.w * im.h;
#
#     for (int i = 0; i < c_size; i++)
#     {
#         float H = im.data[i] * 6;
#         float S = im.data[i+c_size];
#         float V = im.data[i+2*c_size];
#
#         float C = V * S;
#         float X = C * (1-fabs(fmod(H, 2) - 1));
#         float m = V-C;
#
#         if (H >= 5) {
#             im.data[i] = (C + m);
#             im.data[i+c_size] = m;
#             im.data[i+2*c_size] = (X + m);
#         } else if (H >= 4) {
#             im.data[i] = (X + m);
#             im.data[i+c_size] = m;
#             im.data[i+2*c_size] = (C + m);
#         } else if (H >= 3) {
#             im.data[i] = m;
#             im.data[i+c_size] = (X + m);
#             im.data[i+2*c_size] = (C + m);
#         } else if (H >= 2) {
#             im.data[i] = m;
#             im.data[i+c_size] = (C + m);
#             im.data[i+2*c_size] = (X + m);
#         } else if (H >= 1) {
#             im.data[i] = (X + m);
#             im.data[i+c_size] = (C + m);
#             im.data[i+2*c_size] = m;
#         } else {
#             im.data[i] = (C + m);
#             im.data[i+c_size] = (X + m);
#             im.data[i+2*c_size] = m;
#         }
#     }
# }
#
# void scale_image(image im, int c, float v)
# {
#     // TODO Fill this in
#     int c_size = im.w * im.h;
#
#     for (int i = c_size * c; i < c_size * c + c_size; i++)
#     {
#         im.data[i] *= v;
#     }
# }

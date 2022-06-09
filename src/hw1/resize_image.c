#include <math.h>
#include "image.h"

float nn_interpolate(image im, float x, float y, int c)
{
    // TODO Fill in
    return get_pixel(im, round(x), round(y), c);
}

image nn_resize(image im, int w, int h)
{
    // TODO Fill in (also fix that first line)
    float prev_w = (float) im.w;
    float prev_h = (float) im.h;
    int c = im.c;
    image new_image = make_image(w,h,im.c);
    for (int i = 0; i < w; i++)
    {
        for (int j = 0; j < h; j++)
        {
            for (int k = 0; k < c; k++)
            {
                float i1 = prev_w/w * i + 0.5 * (prev_w/w - 1);
                float j1 = prev_h/h * j + 0.5 * (prev_h/h - 1);
                set_pixel(new_image, i, j, k, nn_interpolate(im, i1, j1, k));
            }
        }
    }

    return new_image;
}

float bilinear_interpolate(image im, float x, float y, int c)
{
    // TODO
    // floor x, ceiling x, etc.
    int fx = floor(x);
    int cx = fx + 1;
    int fy = floor(y);
    int cy = fy + 1;

    float fxfy = get_pixel(im, fx, fy, c);
    float fxcy = get_pixel(im, fx, cy, c);
    float cxfy = get_pixel(im, cx, fy, c);
    float cxcy = get_pixel(im, cx, cy, c);

    float l_interpolate = fxfy * (cy-y) + fxcy * (y-fy);
    float r_interpolate = cxfy * (cy-y) + cxcy * (y-fy);

    return l_interpolate * (cx-x) + r_interpolate * (x-fx);
}

image bilinear_resize(image im, int w, int h)
{
    // TODO
    float prev_w = (float) im.w;
    float prev_h = (float) im.h;
    int c = im.c;
    image new_image = make_image(w,h,im.c);
    for (int i = 0; i < w; i++)
    {
        for (int j = 0; j < h; j++)
        {
            for (int k = 0; k < c; k++)
            {
                float i1 = prev_w/w * i + 0.5 * (prev_w/w - 1);
                float j1 = prev_h/h * j + 0.5 * (prev_h/h - 1);
                set_pixel(new_image, i, j, k, bilinear_interpolate(im, i1, j1, k));
            }
        }
    }

    return new_image;
}


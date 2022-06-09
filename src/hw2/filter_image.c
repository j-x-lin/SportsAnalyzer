#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "image.h"
#define TWOPI 6.2831853

void l1_normalize(image im)
{
    // TODO
    int w = im.w;
    int h = im.h;
    int c = im.c;

    float total = 0;
    for (int i = 0; i < w * h * c; i++)
    {
        total += fabs(im.data[i]);
    }

    for (int i = 0; i < w * h * c; i++)
    {
        im.data[i] /= total;
    }
}

image make_box_filter(int w)
{
    // TODO
    image box_filter = make_image(w,w,1);
    for (int i = 0; i < w*w; i++)
    {
        box_filter.data[i] = 1;
    }

    l1_normalize(box_filter);
    return box_filter;
}

static float convolve_pixel(image im, image filter, int x, int y, int im_c, int fil_c)
{
    float val = 0;
    for (int i = 0; i < filter.w; i++)
    {
        for (int j = 0; j < filter.h; j++)
        {
            int im_x = x+i-filter.w/2;
            int im_y = y+j-filter.h/2;
            float fil_val = get_pixel(filter, i, j, fil_c);
            val += get_pixel(im, im_x, im_y, im_c) * fil_val;
        }
    }

    return val;
}

image convolve_image(image im, image filter, int preserve)
{
    // TODO
    image new_image;
    
    if (filter.c == 1) {
        if (preserve == 1) {
            // apply filter separately to each channel
            new_image = make_image(im.w, im.h, im.c);
            for (int i = 0; i < im.w; i++)
            {
                for (int j = 0; j < im.h; j++)
                {
                    for (int k = 0; k < im.c; k++)
                    {
                        float val = convolve_pixel(im, filter, i, j, k, 0);
                        set_pixel(new_image, i, j, k, val);
                    }
                }
            }
        } else {
            // apply filter to each channel, then sum
            new_image = make_image(im.w, im.h, 1);
            for (int i = 0; i < im.w; i++)
            {
                for (int j = 0; j < im.h; j++)
                {
                    float val = 0;

                    for (int k = 0; k < im.c; k++)
                    {
                        val += convolve_pixel(im, filter, i, j, k, 0);
                    }

                    set_pixel(new_image, i, j, 0, val);
                }
            }
        }
    } else if (filter.c == im.c) {
        if (preserve == 1) {
            // apply filter, keeping each channel separate
            new_image = make_image(im.w, im.h, im.c);
            for (int i = 0; i < im.w; i++)
            {
                for (int j = 0; j < im.h; j++)
                {
                    for (int k = 0; k < im.c; k++)
                    {
                        float val = convolve_pixel(im, filter, i, j, k, k);
                        set_pixel(new_image, i, j, k, val);
                    }
                }
            }
            
        } else {
            // apply filter, then sum
            new_image = make_image(im.w, im.h, im.c);
            for (int i = 0; i < im.w; i++)
            {
                for (int j = 0; j < im.h; j++)
                {
                    float val = 0;
                    for (int k = 0; k < im.c; k++)
                    {
                        val += convolve_pixel(im, filter, i, j, k, k);
                    }

                    set_pixel(new_image, i, j, 0, val);
                }
            }
        }
    } else {
        new_image = make_image(1, 1, 1);
    }

    return new_image;
}

image make_highpass_filter()
{
    // TODO
    image hp_filter = make_image(3, 3, 1);
    set_pixel(hp_filter, 0, 0, 0, 0);
    set_pixel(hp_filter, 0, 1, 0, -1);
    set_pixel(hp_filter, 0, 2, 0, 0);
    set_pixel(hp_filter, 1, 0, 0, -1);
    set_pixel(hp_filter, 1, 1, 0, 4);
    set_pixel(hp_filter, 1, 2, 0, -1);
    set_pixel(hp_filter, 2, 0, 0, 0);
    set_pixel(hp_filter, 2, 1, 0, -1);
    set_pixel(hp_filter, 2, 2, 0, 0);
    return hp_filter;
}

image make_sharpen_filter()
{
    // TODO
    image sharpen_filter = make_image(3, 3, 1);
    set_pixel(sharpen_filter, 0, 0, 0, 0);
    set_pixel(sharpen_filter, 0, 1, 0, -1);
    set_pixel(sharpen_filter, 0, 2, 0, 0);
    set_pixel(sharpen_filter, 1, 0, 0, -1);
    set_pixel(sharpen_filter, 1, 1, 0, 5);
    set_pixel(sharpen_filter, 1, 2, 0, -1);
    set_pixel(sharpen_filter, 2, 0, 0, 0);
    set_pixel(sharpen_filter, 2, 1, 0, -1);
    set_pixel(sharpen_filter, 2, 2, 0, 0);
    return sharpen_filter;
}

image make_emboss_filter()
{
    // TODO
    image boss_filter = make_image(3, 3, 1);
    set_pixel(boss_filter, 0, 0, 0, -2);
    set_pixel(boss_filter, 0, 1, 0, -1);
    set_pixel(boss_filter, 0, 2, 0, 0);
    set_pixel(boss_filter, 1, 0, 0, -1);
    set_pixel(boss_filter, 1, 1, 0, 1);
    set_pixel(boss_filter, 1, 2, 0, 1);
    set_pixel(boss_filter, 2, 0, 0, 0);
    set_pixel(boss_filter, 2, 1, 0, 1);
    set_pixel(boss_filter, 2, 2, 0, 2);
    return boss_filter;
}

// Question 2.2.1: Which of these filters should we use preserve when we run our convolution and which ones should we not? Why?
// Answer: we should use preserve on the emboss and sharpen filters since we ultimately want to turn them into RGB images.
// However, the high-pass filter just looks for high-frequency regions, and we usually desire grayscale images showing the result.

// Question 2.2.2: Do we have to do any post-processing for the above filters? Which ones and why?
// Answer: The sharpen and emboss filters we may have to do some clamping, since they could in some cases give us overflow.

image make_gaussian_filter(float sigma)
{
    // TODO
    int size = ((int) (3 * sigma)) * 2 + 1;
    image gauss_filter = make_image(size, size, 1);

    for (int j = 0; j < size; j++)
    {
        for (int i = 0; i < size; i++)
        {
            int x = i - size/2;
            int y = j - size/2;

            float val = 1/(TWOPI*sigma*sigma);
            val *= exp(-(x*x + y*y) / (2*sigma*sigma));
            set_pixel(gauss_filter, i, j, 0, val);
        }
    }

    l1_normalize(gauss_filter);
    return gauss_filter;
}

image add_image(image a, image b)
{
    // TODO
    if (a.w == b.w && a.h == b.h && a.c == b.c) {
        image sum = make_image(a.w, a.h, a.c);

        for (int i = 0; i < a.w*a.h*a.c; i++)
        {
            sum.data[i] = a.data[i] + b.data[i];
        }

        return sum;
    }
    
    return make_image(1,1,1);
}

image sub_image(image a, image b)
{
    // TODO
    if (a.w == b.w && a.h == b.h && a.c == b.c) {
        image diff = make_image(a.w, a.h, a.c);

        for (int i = 0; i < a.w*a.h*a.c; i++)
        {
            diff.data[i] = a.data[i] - b.data[i];
        }

        return diff;
    }

    return make_image(1, 1, 1);
}

image make_gx_filter()
{
    // TODO
    image gx = make_image(3,3,1);
    gx.data[0] = -1;
    gx.data[1] = 0;
    gx.data[2] = 1;
    gx.data[3] = -2;
    gx.data[4] = 0;
    gx.data[5] = 2;
    gx.data[6] = -1;
    gx.data[7] = 0;
    gx.data[8] = 1;
    return gx;
}

image make_gy_filter()
{
    // TODO
    image gy = make_image(3,3,1);
    gy.data[0] = -1;
    gy.data[1] = -2;
    gy.data[2] = -1;
    gy.data[3] = 0;
    gy.data[4] = 0;
    gy.data[5] = 0;
    gy.data[6] = 1;
    gy.data[7] = 2;
    gy.data[8] = 1;
    return gy;
}


void feature_normalize(image im)
{
    // TODO
    float max = 0;
    float min = 1;
    for (int i = 0; i < im.w * im.h * im.c; i++)
    {
        if (im.data[i] < min) {
            min = im.data[i];
        }

        if (im.data[i] > max) {
            max = im.data[i];
        }
    }

    for (int i = 0; i < im.w * im.h * im.c; i++)
    {
        if (min != max) {
            im.data[i] = (im.data[i] - min) / (max - min);
        } else {
            im.data[i] = 0;
        }
    }
}

image *sobel_image(image im)
{
    // TODO
    image* magdir = (image*) calloc(2, sizeof(image));
    image gx = make_gx_filter();
    image gy = make_gy_filter();

    image mag = make_image(im.w, im.h, 1);
    image dir = make_image(im.w, im.h, 1);

    magdir[0] = mag;
    magdir[1] = dir;

    image sobelx = convolve_image(im, gx, 0);
    image sobely = convolve_image(im, gy, 0);

    for (int i = 0; i < im.w * im.h; i++)
    {
        float X = sobelx.data[i];
        float Y = sobely.data[i];

        mag.data[i] = sqrt(X * X + Y * Y);
        dir.data[i] = atan2(X, Y);
    }

    return magdir;
}

image colorize_sobel(image im)
{
    // TODO
    image* magdir = sobel_image(im);
    image mag = magdir[0];
    image dir = magdir[1];

    image cool_image = make_image(mag.w, mag.h, 3);
    int channel = im.w * im.h;

    for (int i = 0; i < channel; i++)
    {
        cool_image.data[i] = dir.data[i];
        cool_image.data[i+channel] = mag.data[i];
        cool_image.data[i+2*channel] = mag.data[i];
    }

    hsv_to_rgb(cool_image);
    return cool_image;
}

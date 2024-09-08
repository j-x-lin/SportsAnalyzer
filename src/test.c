#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include "matrix.h"
#include "image.h"
#include "test.h"
#include "args.h"


float avg_diff(image a, image b)
{
    float diff = 0;
    int i;
    for(i = 0; i < a.w*a.h*a.c; ++i){
        diff += b.data[i] - a.data[i];
    }
    return diff/(a.w*a.h*a.c);
}

image center_crop(image im)
{
    image c = make_image(im.w/2, im.h/2, im.c);
    int i, j, k;
    for(k = 0; k < im.c; ++k){
        for(j = 0; j < im.h/2; ++j){
            for(i = 0; i < im.w/2; ++i){
                set_pixel(c, i, j, k, get_pixel(im, i+im.w/4, j+im.h/4, k));
            }
        }
    }
    return c;
}

void feature_normalize2(image im)
{
    int i;
    if (!im.data) return;
    float min = im.data[0];
    float max = im.data[0];
    for(i = 0; i < im.w*im.h*im.c; ++i){
        if(im.data[i] > max) max = im.data[i];
        if(im.data[i] < min) min = im.data[i];
    }
    for(i = 0; i < im.w*im.h*im.c; ++i){
        im.data[i] = (im.data[i] - min)/(max-min);
    }
}

int tests_total = 0;
int tests_fail = 0;

int within_eps(float a, float b, float eps){
    int res = a-eps<b && b<a+eps;
    return res;
}

int same_point(point p, point q, float eps)
{
    return within_eps(p.x, q.x, eps) && within_eps(p.y, q.y, eps);
}

int same_matrix(matrix m, matrix n)
{
    if(m.rows != n.rows || m.cols != n.cols) return 0;
    int i,j;
    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            if(!within_eps(m.data[i][j], n.data[i][j], EPS)) {
                printf("%f but was %f at %d %d\n", m.data[i][j], n.data[i][j], i, j);
                return 0;
            }
        }
    }
    return 1;
}

int same_image(image a, image b, float eps)
{
    int i;
    if(a.w != b.w || a.h != b.h || a.c != b.c) {
        printf("    Expected %d x %d x %d image, got %d x %d x %d\n", b.w, b.h, b.c, a.w, a.h, a.c);
        return 0;
    }
    for(i = 0; i < a.w*a.h*a.c; ++i){
        int x = i % a.w;
        int y = (i / a.w) % a.h;
        int z = (i / (a.w * a.h));
        float thresh = (fabs(b.data[i]) + fabs(a.data[i])) * eps / 2;
        if (thresh > eps) eps = thresh;
        if(!within_eps(a.data[i], b.data[i], eps)) 
        {
            printf("    Index %d, Pixel (%d, %d, %d) should be %f, but it is %f! \n", i, x, y, z, b.data[i], a.data[i]);
            return 0;
        }
    }
    return 1;
}

void make_hw0_test()
{
    image dots = make_image(4, 2, 3);
    set_pixel(dots, 0, 0, 0, 0/255.);
    set_pixel(dots, 0, 0, 1, 1/255.);
    set_pixel(dots, 0, 0, 2, 2/255.);
    
    set_pixel(dots, 1, 0, 0, 255/255.);
    set_pixel(dots, 1, 0, 1, 3/255.);
    set_pixel(dots, 1, 0, 2, 4/255.);
    
    set_pixel(dots, 2, 0, 0, 5/255.);
    set_pixel(dots, 2, 0, 1, 254/255.);
    set_pixel(dots, 2, 0, 2, 6/255.);
    
    set_pixel(dots, 3, 0, 0, 7/255.);
    set_pixel(dots, 3, 0, 1, 8/255.);
    set_pixel(dots, 3, 0, 2, 253/255.);

    set_pixel(dots, 0, 1, 0, 252/255.);
    set_pixel(dots, 0, 1, 1, 251/255.);
    set_pixel(dots, 0, 1, 2, 250/255.);
    
    set_pixel(dots, 1, 1, 0, 9/255.);
    set_pixel(dots, 1, 1, 1, 249/255.);
    set_pixel(dots, 1, 1, 2, 248/255.);
    
    set_pixel(dots, 2, 1, 0, 247/255.);
    set_pixel(dots, 2, 1, 1, 10/255.);
    set_pixel(dots, 2, 1, 2, 246/255.);
    
    set_pixel(dots, 3, 1, 0, 245/255.);
    set_pixel(dots, 3, 1, 1, 244/255.);
    set_pixel(dots, 3, 1, 2, 11/255.);
    save_png(dots, "data/dotz");
}

void make_matrix_test()
{
    srand(1);
    matrix a = random_matrix(32, 64, 10);
    matrix w = random_matrix(64, 16, 10);
    matrix y = random_matrix(32, 64, 10);
    matrix dw = random_matrix(64, 16, 10);
    matrix v = random_matrix(64, 16, 10);
    matrix delta = random_matrix(32, 16, 10);

    save_matrix(a, "data/test/a.matrix");
    save_matrix(w, "data/test/w.matrix");
    save_matrix(dw, "data/test/dw.matrix");
    save_matrix(v, "data/test/v.matrix");
    save_matrix(delta, "data/test/delta.matrix");
    save_matrix(y, "data/test/y.matrix");

    matrix alog = copy_matrix(a);
    activate_matrix(alog, LOGISTIC);
    save_matrix(alog, "data/test/alog.matrix");

    matrix arelu = copy_matrix(a);
    activate_matrix(arelu, RELU);
    save_matrix(arelu, "data/test/arelu.matrix");

    matrix alrelu = copy_matrix(a);
    activate_matrix(alrelu, LRELU);
    save_matrix(alrelu, "data/test/alrelu.matrix");

    matrix asoft = copy_matrix(a);
    activate_matrix(asoft, SOFTMAX);
    save_matrix(asoft, "data/test/asoft.matrix");


    matrix glog = copy_matrix(a);
    gradient_matrix(y, LOGISTIC, glog);
    save_matrix(glog, "data/test/glog.matrix");

    matrix grelu = copy_matrix(a);
    gradient_matrix(y, RELU, grelu);
    save_matrix(grelu, "data/test/grelu.matrix");

    matrix glrelu = copy_matrix(a);
    gradient_matrix(y, LRELU, glrelu);
    save_matrix(glrelu, "data/test/glrelu.matrix");

    matrix gsoft = copy_matrix(a);
    gradient_matrix(y, SOFTMAX, gsoft);
    save_matrix(gsoft, "data/test/gsoft.matrix");


    layer l = make_layer(64, 16, LRELU);
    l.w = w;
    l.dw = dw;
    l.v = v;

    matrix out = forward_layer(&l, a);
    save_matrix(out, "data/test/out.matrix");

    matrix dx = backward_layer(&l, delta);
    save_matrix(l.dw, "data/test/truth_dw.matrix");
    save_matrix(l.v, "data/test/truth_v.matrix");
    save_matrix(dx, "data/test/truth_dx.matrix");

    update_layer(&l, .01, .9, .01);
    save_matrix(l.dw, "data/test/updated_dw.matrix");
    save_matrix(l.w, "data/test/updated_w.matrix");
    save_matrix(l.v, "data/test/updated_v.matrix");
}

void test_hw0()
{
    printf("%d tests, %d passed, %d failed\n", tests_total, tests_total-tests_fail, tests_fail);
}
void test_hw1()
{
    printf("%d tests, %d passed, %d failed\n", tests_total, tests_total-tests_fail, tests_fail);
}
void test_hw2()
{
    printf("%d tests, %d passed, %d failed\n", tests_total, tests_total-tests_fail, tests_fail);
}
void test_hw3()
{
    printf("%d tests, %d passed, %d failed\n", tests_total, tests_total-tests_fail, tests_fail);
}
void make_hw4_tests()
{
    image dots = load_image("data/dots.png");
    image intdot = make_integral_image(dots);
    save_image_binary(intdot, "data/dotsintegral.bin");

    image dogbw = load_image("data/dogbw.png");
    image intdog = make_integral_image(dogbw);
    save_image_binary(intdog, "data/dogintegral.bin");

    image dog = load_image("data/dog.jpg");
    image smooth = box_filter_image(dog, 15);
    save_png(smooth, "data/dogbox");

    image smooth_c = center_crop(smooth);
    save_png(smooth_c, "data/dogboxcenter");

    image doga = load_image("data/dog_a_small.jpg");
    image dogb = load_image("data/dog_b_small.jpg");
    image structure = time_structure_matrix(dogb, doga, 15);
    save_image_binary(structure, "data/structure.bin");

    image velocity = velocity_image(structure, 5);
    save_image_binary(velocity, "data/velocity.bin");
}
void test_hw4()
{
    printf("%d tests, %d passed, %d failed\n", tests_total, tests_total-tests_fail, tests_fail);
}
void test_hw5()
{
    printf("%d tests, %d passed, %d failed\n", tests_total, tests_total-tests_fail, tests_fail);
}

void run_tests()
{
    printf("%d tests, %d passed, %d failed\n", tests_total, tests_total-tests_fail, tests_fail);
}


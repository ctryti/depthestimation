__constant const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

// Defined values used in this file:
// #define WIDTH = input width
// #define THRESHOLD = some small value, ~2



__kernel void crossCheck(const __global uchar *left_result,
                         const __global uchar *right_result,
                         __global uchar *new_left,
                         __global uchar *new_right) {

    const int2 gid;
    gid.x = get_global_id(0);
    gid.y = get_global_id(1);

    if(gid.x < WIDTH) {

        int left_disparity = left_result[gid.y * WIDTH + gid.x];
        int right_disparity = right_result[gid.y * WIDTH + gid.x - left_disparity];

        if(abs(left_disparity - right_disparity) > THRESHOLD) {
            new_left[gid.y * WIDTH + gid.x] = 0;
        } else {
            new_left[gid.y * WIDTH + gid.x] = left_result[gid.y * WIDTH + gid.x];
        }

        right_disparity = right_result[gid.y * WIDTH + gid.x];
        left_disparity = left_result[gid.y * WIDTH + gid.x + right_disparity];

        if(abs(right_disparity - left_disparity) > THRESHOLD) {
            new_right[gid.y * WIDTH + gid.x] = 0;
        } else {
            new_right[gid.y * WIDTH + gid.x] = right_result[gid.y * WIDTH + gid.x];
        }
    }
}

__kernel void crossCheck_image2d(__read_only image2d_t input_l,
                                 __read_only image2d_t input_r,
                                 __write_only image2d_t output_l,
                                 __write_only image2d_t output_r) {

    const int2 gid;
    int2 cross_coord;
    gid.x = get_global_id(0);
    gid.y = get_global_id(1);

    const int width = get_image_width(input_l);

    if(gid.x < width) {

        cross_coord.y = gid.y;

        int left_disparity  = read_imageui(input_l, sampler, gid).x;
        cross_coord.x = gid.x - left_disparity;
        int right_disparity = read_imageui(input_r, sampler, cross_coord).x;

        /* if the result is below THRESHOLD, write a 0. Else write the disparity */
        uchar checked_res = step((float)abs(left_disparity - right_disparity), (float)THRESHOLD) * left_disparity;
        write_imageui(output_l, gid, checked_res);

        right_disparity = read_imageui(input_r, sampler, gid).x;
        cross_coord.x = gid.x + right_disparity;
        left_disparity = read_imageui(input_l, sampler, cross_coord).x;

        checked_res = step((float)abs(right_disparity - left_disparity), (float)THRESHOLD) * right_disparity;
        write_imageui(output_r, gid, checked_res);
    }
}

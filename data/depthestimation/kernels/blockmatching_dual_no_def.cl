/*
 * Block matching, uses local memory, produces left and right
 * disparity map 817
 */
__kernel void calculateDisparityDual_no_def(const __global uchar *img_l,
                                     const __global uchar *img_r,
                                     __global uchar *result_l,
                                     __global uchar *result_r,
                                     const __global uchar *prev_result_l,
                                     const __global uchar *prev_result_r,
                                     __local uchar *local_left,
                                     __local uchar *local_right,
                                     const int local_width,
                                     const int local_height,
                                     __global uchar *diff_map,
                                     const int width,
                                     const int height,
                                     const int padded_width,
                                     const int max_disp,
                                     const int aggr_radius,
                                     const int aggr_dim
                                    ) {

    int2 gid;
    int2 lid;
    gid.x = get_global_id(0);
    gid.y = get_global_id(1);
    lid.x = get_local_id(0);
    lid.y = get_local_id(1);

    /* if(diff_map[gid.y * WIDTH + gid.x] == 0) { */
    /*     result_l[gid.y * WIDTH + gid.x] = prev_result_l[gid.y * WIDTH + gid.x]; */
    /*     result_r[gid.y * WIDTH + gid.x] = prev_result_r[gid.y * WIDTH + gid.x]; */
    /*     return; */
    /* } */


    int current_right_sum = 0;
    int current_left_sum = 0;
    uint best_left_sum = -1;
    uint best_right_sum = -1;
    uchar best_left_disparity = 0;
    uchar best_right_disparity = 0;

    int i = 0;
    int j = 0;
    int d = 0;

    /* fill the left local memory buffer */
    for(i = 0; i < local_height; i += get_local_size(1)) {
        for(j = 0; j < local_width; j += get_local_size(0)) {
            local_left[(lid.y + i) * local_width + (lid.x + j)] =
                img_l[((gid.y - aggr_radius + i) * padded_width) + gid.x - aggr_radius + j];

            local_right[(lid.y + i) * local_width + (lid.x + j)] =
                img_r[((gid.y - aggr_radius + i) * padded_width) + gid.x - aggr_radius + j - max_disp];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    /* find the best match */
    for(d = 0; d < max_disp; d++) {
        for(i = 0; i < aggr_dim; i++) {
            for(j = 0; j < aggr_dim; j++) {
                current_left_sum += abs(local_left[((lid.y+i) * local_width) + lid.x + j] -
                                        local_right[((lid.y+i) * local_width) + lid.x + j - d + max_disp]);

                /* current_right_sum += abs(local_right[((lid.y+i) * local_width) + lid.x + j + max_disp] - */
                /*                          local_left[((lid.y+i) * local_width) + lid.x + j + d]); */
            }
        }

        if(current_left_sum < best_left_sum) {
            best_left_sum = current_left_sum;
            best_left_disparity = d;
        }
        current_left_sum = 0;

        /* if(current_right_sum < best_right_sum) { */
        /*     best_right_sum = current_right_sum; */
        /*     best_right_disparity = d; */
        /* } */
        /* current_right_sum = 0; */

    }

    /* find the best match */
    for(d = 0; d < max_disp; d++) {
        for(i = 0; i < aggr_dim; i++) {
            for(j = 0; j < aggr_dim; j++) {
                current_right_sum += abs(local_right[((lid.y+i) * local_width) + lid.x + j + max_disp] -
                                         local_left[((lid.y+i) * local_width) + lid.x + j + d]);
            }
        }

        if(current_right_sum < best_right_sum) {
            best_right_sum = current_right_sum;
            best_right_disparity = d;
        }
        current_right_sum = 0;

    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(gid.x < width && gid.y < height) {
        result_l[gid.y * width + gid.x] = best_left_disparity;
        result_r[gid.y * width + gid.x] = best_right_disparity;
    }
}

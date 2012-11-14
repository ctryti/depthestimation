/*
 * Block matching, uses local memory, produces left and right
 * disparity map 817
 */
__kernel void calculateDisparityDual(const __global uchar *img_l,
                                     const __global uchar *img_r,
                                     __global uchar *result_l,
                                     __global uchar *result_r,
                                     const __global uchar *prev_result_l,
                                     const __global uchar *prev_result_r,
                                     __local uchar *local_left,
                                     __local uchar *local_right,
                                     const int local_width,
                                     const int local_height,
                                     __global uchar *diff_map
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
                img_l[((gid.y - AGGR_RADIUS + i) * PADDED_WIDTH) + gid.x - AGGR_RADIUS + j];

            local_right[(lid.y + i) * local_width + (lid.x + j)] =
                img_r[((gid.y - AGGR_RADIUS + i) * PADDED_WIDTH) + gid.x - AGGR_RADIUS + j - MAX_DISP];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    /* find the best match */
    for(d = 0; d < MAX_DISP; d++) {
        for(i = 0; i < AGGR_DIM; i++) {
            for(j = 0; j < AGGR_DIM; j++) {
                current_left_sum += abs(local_left[((lid.y+i) * local_width) + lid.x + j] -
                                        local_right[((lid.y+i) * local_width) + lid.x + j - d + MAX_DISP]);

                /* current_right_sum += abs(local_right[((lid.y+i) * local_width) + lid.x + j + MAX_DISP] - */
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
    for(d = 0; d < MAX_DISP; d++) {
        for(i = 0; i < AGGR_DIM; i++) {
            for(j = 0; j < AGGR_DIM; j++) {
                current_right_sum += abs(local_right[((lid.y+i) * local_width) + lid.x + j + MAX_DISP] -
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

    if(gid.x < WIDTH && gid.y < HEIGHT) {
        result_l[gid.y * WIDTH + gid.x] = best_left_disparity;
        result_r[gid.y * WIDTH + gid.x] = best_right_disparity;
    }
}

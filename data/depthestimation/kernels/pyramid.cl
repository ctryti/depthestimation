__constant const sampler_t sampler            = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;
__constant const sampler_t normalized_sampler = CLK_NORMALIZED_COORDS_TRUE  | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

uchar calculateDisparity(const int mip_lvl,
                         const int min_d,
                         const int max_d,
                         const int aggr_dim,
                         const int width,
                         __local uchar *input_l,
                         __local uchar *input_r
                         ) {

    int max_disp = MAX_DISP/mip_lvl;
    int d, i, j;

    int sum;
    uint best_sum = -1;
    int best_d = 0;

    int2 lid = (int2)(get_local_id(0), get_local_id(1));

    /* find the best match */
    for(d = min_d; d < max_d; d++) {
        for(i = 0; i < aggr_dim; i++) {
            for(j = 0; j < aggr_dim; j++) {
                sum += abs(input_l[((lid.y+i) * width) + lid.x + j] -
                           input_r[((lid.y+i) * width) + lid.x + j - d + max_disp]);
            }
        }
        if(sum < best_sum) {
            best_sum = sum;
            best_d = d;
        }
        sum = 0;

    }
    barrier(CLK_LOCAL_MEM_FENCE);

    return best_d;
}

void findMinMax(int mip_lvl,
                __read_only image2d_t input_l,
                __read_only image2d_t input_r,
                __local uchar *local_l,
                __local uchar *local_r,
                int *min_d_l,
                int *max_d_l,
                int *min_d_r,
                int *max_d_r) {

    int2 gid;
    int2 lid;
    gid.x = get_global_id(0);
    gid.y = get_global_id(1);
    lid.x = get_local_id(0);
    lid.y = get_local_id(1);

    int tid = (lid.y * get_local_size(0)) + lid.x;
    int group_size = get_local_size(0) * get_local_size(1);

    // float2 prev_res_coords = (float2) ((float) gid.x / (WIDTH/mip_lvl), (float) gid.y / (HEIGHT/mip_lvl));
    // local_l[tid] = (uchar)read_imageui(input_l, normalized_sampler, prev_res_coords).x;
    // local_r[tid] = (uchar)read_imageui(input_r, normalized_sampler, prev_res_coords).x;
    int2 prev_res_coords = (int2) (gid.x / 2, gid.y / 2);
    local_l[tid] = (uchar)read_imageui(input_l, sampler, prev_res_coords).x;
    local_r[tid] = (uchar)read_imageui(input_r, sampler, prev_res_coords).x;
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

#ifdef USE_MIN_MAX
    
    int cutoff = group_size/2;
    int step = group_size;
    do {
        step >>= 1;
        int idx = tid + (step * (tid/step));

        if(tid < cutoff) {
            uchar tmp = local_l[idx];
            if(tmp > local_l[idx + step]) {
                local_l[idx] = local_l[idx + step];
                local_l[idx + step] = tmp;
            }
            tmp = local_r[idx];
            if(tmp > local_r[idx + step]) {
                local_r[idx] = local_r[idx + step];
                local_r[idx + step] = tmp;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    } while(step > 1);

    *min_d_l = local_l[0];
    *min_d_l *= 2;
    *min_d_l -= 2;

    *max_d_l = local_l[group_size - 1];
    *max_d_l *= 2;
    *max_d_l += 2;

    *min_d_r = local_r[0];
    *min_d_r *= 2;
    *min_d_r -= 2;

    *max_d_r = local_r[group_size - 1];
    *max_d_r *= 2;
    *max_d_r += 2;

#else
    
    *min_d_l = local_l[tid];
    *min_d_l *= 2;
    *min_d_l -= 2;

    *max_d_l = local_l[tid];
    *max_d_l *= 2;
    *max_d_l += 2;

    *min_d_r = local_r[tid];
    *min_d_r *= 2;
    *min_d_r -= 2;

    *max_d_r = local_r[tid];
    *max_d_r *= 2;
    *max_d_r += 2;
#endif

    int max_d_for_this_level = MAX_DISP / mip_lvl;
    
    *min_d_l = clamp(*min_d_l, 0, max_d_for_this_level);
    *max_d_l = clamp(*max_d_l, 0, max_d_for_this_level);
    *min_d_r = clamp(*min_d_r, 0, max_d_for_this_level);
    *max_d_r = clamp(*max_d_r, 0, max_d_for_this_level);

    // if(*min_d_l == 0)
    //     max_d_l = max_d_for_this_level;
    // if(*min_d_r == 0)
    //     max_d_r = max_d_for_this_level;

    
    barrier(CLK_LOCAL_MEM_FENCE);
}

__kernel void pyramid0(__read_only image2d_t input_l,
                       __read_only image2d_t input_r,
                       __read_only image2d_t prev_res_l,
                       __read_only image2d_t prev_res_r,
                       __write_only image2d_t output_l,
                       __write_only image2d_t output_r,
                       __local uchar *local_l,
                       __local uchar *local_r,
                       const int local_width,
                       const int local_height) {

    int2 gid;
    int2 lid;
    gid.x = get_global_id(0);
    gid.y = get_global_id(1);
    lid.x = get_local_id(0);
    lid.y = get_local_id(1);

    int current_right_sum = 0;
    int current_left_sum = 0;
    uint best_left_sum = -1;
    uint best_right_sum = -1;
    uchar best_left_disparity = 0;
    uchar best_right_disparity = 0;

    int i = 0;
    int j = 0;
    int d = 0;

    int2 coords;
    uchar pixel;

    int max_d_l;
    int min_d_l;
    int max_d_r;
    int min_d_r;

    findMinMax(1,
               prev_res_l, prev_res_r,
               local_l, local_r,
               &min_d_l, &max_d_l,
               &min_d_r, &max_d_r);

    /* fill the left local memory buffer */
    for(i = 0; i < local_height; i += get_local_size(1)) {
        coords.y = gid.y - AGGR_RADIUS + i;
        for(j = 0; j < local_width; j += get_local_size(0)) {
            coords.x = gid.x - AGGR_RADIUS + j;
            pixel = (uchar)read_imageui(input_l, sampler, coords).x;
            local_l[(lid.y + i) * local_width + (lid.x + j)] = pixel;

            coords.x -= MAX_DISP;
            pixel = (uchar)read_imageui(input_r, sampler, coords).x;
            local_r[(lid.y + i) * local_width + (lid.x + j)] = pixel;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* find the best match */
    for(d = min_d_l; d < max_d_l; d++) {
        for(i = 0; i < AGGR_DIM; i++) {
            for(j = 0; j < AGGR_DIM; j++) {
                current_left_sum += abs(local_l[((lid.y+i) * local_width) + lid.x + j] -
                                        local_r[((lid.y+i) * local_width) + lid.x + j - d + MAX_DISP]);
            }
        }
        if(current_left_sum < best_left_sum) {
            best_left_sum = current_left_sum;
            best_left_disparity = d;
        }
        current_left_sum = 0;

    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* find the best match */
    for(d = min_d_r; d < max_d_r; d++) {
        for(i = 0; i < AGGR_DIM; i++) {
            for(j = 0; j < AGGR_DIM; j++) {
                current_right_sum += abs(local_r[((lid.y+i) * local_width) + lid.x + j + MAX_DISP] -
                                         local_l[((lid.y+i) * local_width) + lid.x + j + d]);
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
        write_imageui(output_l, gid, best_left_disparity);
        write_imageui(output_r, gid, best_right_disparity);
    }
}


__kernel void pyramid1(__read_only image2d_t input_l,
                       __read_only image2d_t input_r,
                       __read_only image2d_t prev_res_l,
                       __read_only image2d_t prev_res_r,
                       __write_only image2d_t output_l,
                       __write_only image2d_t output_r,
                       __local uchar *local_l,
                       __local uchar *local_r,
                       const int local_width,
                       const int local_height) {

    int2 gid;
    int2 lid;
    gid.x = get_global_id(0);
    gid.y = get_global_id(1);
    lid.x = get_local_id(0);
    lid.y = get_local_id(1);

    int current_right_sum = 0;
    int current_left_sum = 0;
    uint best_left_sum = -1;
    uint best_right_sum = -1;
    uchar best_left_disparity = 0;
    uchar best_right_disparity = 0;

    int i = 0;
    int j = 0;
    int d = 0;

    int2 coords;
    uchar pixel;

    int max_d_l;
    int min_d_l;
    int max_d_r;
    int min_d_r;

    findMinMax(2,
               prev_res_l, prev_res_r,
               local_l, local_r,
               &min_d_l, &max_d_l,
               &min_d_r, &max_d_r);

    /* fill the left local memory buffer */
    for(i = 0; i < local_height; i += get_local_size(1)) {
        coords.y = gid.y - AGGR_RADIUS + i;
        for(j = 0; j < local_width; j += get_local_size(0)) {
            coords.x = gid.x - AGGR_RADIUS + j;
            pixel = (uchar)read_imageui(input_l, sampler, coords).x;
            local_l[(lid.y + i) * local_width + (lid.x + j)] = pixel;

            coords.x -= MAX_DISP_1;
            pixel = (uchar)read_imageui(input_r, sampler, coords).x;
            local_r[(lid.y + i) * local_width + (lid.x + j)] = pixel;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* find the best match */
    for(d = min_d_l; d < max_d_l; d++) {
        for(i = 0; i < AGGR_DIM; i++) {
            for(j = 0; j < AGGR_DIM; j++) {
                current_left_sum += abs(local_l[((lid.y+i) * local_width) + lid.x + j] -
                                        local_r[((lid.y+i) * local_width) + lid.x + j - d + MAX_DISP_1]);
            }
        }
        if(current_left_sum < best_left_sum) {
            best_left_sum = current_left_sum;
            best_left_disparity = d;
        }
        current_left_sum = 0;

    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* find the best match */
    for(d = min_d_r; d < max_d_r; d++) {
        for(i = 0; i < AGGR_DIM; i++) {
            for(j = 0; j < AGGR_DIM; j++) {
                current_right_sum += abs(local_r[((lid.y+i) * local_width) + lid.x + j + MAX_DISP_1] -
                                         local_l[((lid.y+i) * local_width) + lid.x + j + d]);
            }
        }
        if(current_right_sum < best_right_sum) {
            best_right_sum = current_right_sum;
            best_right_disparity = d;
        }
        current_right_sum = 0;

    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(gid.x < WIDTH_1 && gid.y < HEIGHT_1) {
        write_imageui(output_l, gid, best_left_disparity);
        write_imageui(output_r, gid, best_right_disparity);
    }
}


__kernel void pyramid2(__read_only image2d_t input_l,
                       __read_only image2d_t input_r,
                       __read_only image2d_t prev_res_l,
                       __read_only image2d_t prev_res_r,
                       __write_only image2d_t output_l,
                       __write_only image2d_t output_r,
                       __local uchar *local_l,
                       __local uchar *local_r,
                       const int local_width,
                       const int local_height) {

    int2 gid;
    int2 lid;
    gid.x = get_global_id(0);
    gid.y = get_global_id(1);
    lid.x = get_local_id(0);
    lid.y = get_local_id(1);

    int current_right_sum = 0;
    int current_left_sum = 0;
    uint best_left_sum = -1;
    uint best_right_sum = -1;
    uchar best_left_disparity = 0;
    uchar best_right_disparity = 0;

    int i = 0;
    int j = 0;
    int d = 0;

    int2 coords;
    uchar pixel;

    int max_d_l;
    int min_d_l;
    int max_d_r;
    int min_d_r;

    findMinMax(4,
               prev_res_l, prev_res_r,
               local_l, local_r,
               &min_d_l, &max_d_l,
               &min_d_r, &max_d_r);

    /* fill the left local memory buffer */
    for(i = 0; i < local_height; i += get_local_size(1)) {
        coords.y = gid.y - AGGR_RADIUS + i;
        for(j = 0; j < local_width; j += get_local_size(0)) {
            coords.x = gid.x - AGGR_RADIUS + j;
            pixel = (uchar)read_imageui(input_l, sampler, coords).x;
            local_l[(lid.y + i) * local_width + (lid.x + j)] = pixel;

            coords.x -= MAX_DISP_2;
            pixel = (uchar)read_imageui(input_r, sampler, coords).x;
            local_r[(lid.y + i) * local_width + (lid.x + j)] = pixel;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* find the best match */
    for(d = min_d_l; d < max_d_l; d++) {
        for(i = 0; i < AGGR_DIM; i++) {
            for(j = 0; j < AGGR_DIM; j++) {
                current_left_sum += abs(local_l[((lid.y+i) * local_width) + lid.x + j] -
                                        local_r[((lid.y+i) * local_width) + lid.x + j - d + MAX_DISP_2]);
            }
        }
        if(current_left_sum < best_left_sum) {
            best_left_sum = current_left_sum;
            best_left_disparity = d;
        }
        current_left_sum = 0;

    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* find the best match */
    for(d = min_d_r; d < max_d_r; d++) {
        for(i = 0; i < AGGR_DIM; i++) {
            for(j = 0; j < AGGR_DIM; j++) {
                current_right_sum += abs(local_r[((lid.y+i) * local_width) + lid.x + j + MAX_DISP_2] -
                                         local_l[((lid.y+i) * local_width) + lid.x + j + d]);
            }
        }
        if(current_right_sum < best_right_sum) {
            best_right_sum = current_right_sum;
            best_right_disparity = d;
        }
        current_right_sum = 0;

    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(gid.x < WIDTH_2  && gid.y < HEIGHT_2) {
        write_imageui(output_l, gid, best_left_disparity);
        write_imageui(output_r, gid, best_right_disparity);
    }
}



__kernel void pyramid3(__read_only image2d_t input_l,
                       __read_only image2d_t input_r,
                       __write_only image2d_t output_l,
                       __write_only image2d_t output_r,
                       __local uchar *local_l,
                       __local uchar *local_r,
                       const int local_width,
                       const int local_height) {

    int2 gid;
    int2 lid;
    gid.x = get_global_id(0);
    gid.y = get_global_id(1);
    lid.x = get_local_id(0);
    lid.y = get_local_id(1);

    int current_right_sum = 0;
    int current_left_sum = 0;
    uint best_left_sum = -1;
    uint best_right_sum = -1;
    uchar best_left_disparity = 0;
    uchar best_right_disparity = 0;

    int i = 0;
    int j = 0;
    int d = 0;

    int2 coords;
    uchar pixel;

    int max_d_l;
    int min_d_l;
    int max_d_r;
    int min_d_r;

    min_d_l = 0;
    min_d_r = 0;
    max_d_l = MAX_DISP_3;
    max_d_r = MAX_DISP_3;

    /* fill the left local memory buffer */
    for(i = 0; i < local_height; i += get_local_size(1)) {
        coords.y = gid.y - 2 + i;
        for(j = 0; j < local_width; j += get_local_size(0)) {
            coords.x = gid.x - 2 + j;
            pixel = (uchar)read_imageui(input_l, sampler, coords).x;
            local_l[(lid.y + i) * local_width + (lid.x + j)] = pixel;

            coords.x -= MAX_DISP_3;
            pixel = (uchar)read_imageui(input_r, sampler, coords).x;
            local_r[(lid.y + i) * local_width + (lid.x + j)] = pixel;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* find the best match */
    for(d = min_d_l; d < max_d_l; d++) {
        for(i = 0; i < 5; i++) {
            for(j = 0; j < 5; j++) {
                current_left_sum += abs(local_l[((lid.y+i) * local_width) + lid.x + j] -
                                        local_r[((lid.y+i) * local_width) + lid.x + j - d + MAX_DISP_3]);
            }
        }
        if(current_left_sum < best_left_sum) {
            best_left_sum = current_left_sum;
            best_left_disparity = d;
        }
        current_left_sum = 0;

    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* find the best match */
    for(d = min_d_r; d < max_d_r; d++) {
        for(i = 0; i < 5; i++) {
            for(j = 0; j < 5; j++) {
                current_right_sum += abs(local_r[((lid.y+i) * local_width) + lid.x + j + MAX_DISP_3] -
                                         local_l[((lid.y+i) * local_width) + lid.x + j + d]);
            }
        }
        if(current_right_sum < best_right_sum) {
            best_right_sum = current_right_sum;
            best_right_disparity = d;
        }
        current_right_sum = 0;

    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(gid.x < WIDTH_3 && gid.y < HEIGHT_3) {
        write_imageui(output_l, gid, best_left_disparity);
        write_imageui(output_r, gid, best_right_disparity);
    }
}


__kernel void pyramid(const int mip_lvl,
                      __read_only image2d_t input_l,
                      __read_only image2d_t input_r,
                      __read_only image2d_t prev_res_l,
                      __read_only image2d_t prev_res_r,
                      __write_only image2d_t output_l,
                      __write_only image2d_t output_r,
                      __local uchar *local_l,
                      __local uchar *local_r,
                      const int local_width,
                      const int local_height) {

    int2 gid;
    int2 lid;
    gid.x = get_global_id(0);
    gid.y = get_global_id(1);
    lid.x = get_local_id(0);
    lid.y = get_local_id(1);

    int current_right_sum = 0;
    int current_left_sum = 0;
    uint best_left_sum = -1;
    uint best_right_sum = -1;
    uchar best_left_disparity = 0;
    uchar best_right_disparity = 0;

    int i = 0;
    int j = 0;
    int d = 0;

    int2 coords;
    uchar pixel;

    int max_d_l;
    int min_d_l;
    int max_d_r;
    int min_d_r;

    if(mip_lvl == 8) {

        min_d_l = 0;
        min_d_r = 0;
        max_d_l = MAX_DISP/mip_lvl;
        max_d_r = MAX_DISP/mip_lvl;

    } else {

        findMinMax(mip_lvl,
                   prev_res_l, prev_res_r,
                   local_l, local_r,
                   &min_d_l, &max_d_l,
                   &min_d_r, &max_d_r);

    }

    int aggr_radius = (mip_lvl == 8) ? 2 : AGGR_RADIUS;
    int aggr_dim = (mip_lvl == 8) ? 5 : AGGR_DIM;

    /* fill the left local memory buffer */
    for(i = 0; i < local_height; i += get_local_size(1)) {
        coords.y = gid.y - aggr_radius + i;
        for(j = 0; j < local_width; j += get_local_size(0)) {
            coords.x = gid.x - aggr_radius + j;
            pixel = (uchar)read_imageui(input_l, sampler, coords).x;
            local_l[(lid.y + i) * local_width + (lid.x + j)] = pixel;

            coords.x -= (MAX_DISP/mip_lvl);
            pixel = (uchar)read_imageui(input_r, sampler, coords).x;
            local_r[(lid.y + i) * local_width + (lid.x + j)] = pixel;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* find the best match */
    for(d = min_d_l; d < max_d_l; d++) {
        for(i = 0; i < aggr_dim; i++) {
            for(j = 0; j < aggr_dim; j++) {
                current_left_sum += abs(local_l[((lid.y+i) * local_width) + lid.x + j] -
                                        local_r[((lid.y+i) * local_width) + lid.x + j - d + (MAX_DISP/mip_lvl)]);
            }
        }
        if(current_left_sum < best_left_sum) {
            best_left_sum = current_left_sum;
            best_left_disparity = d;
        }
        current_left_sum = 0;

    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    //best_left_disparity = calculateDisparity(mip_lvl, min_d_l, max_d_l, aggr_radius, local_width, local_l, local_r);
    //best_right_disparity = calculateDisparity(mip_lvl, min_d_r, max_d_r, aggr_radius, local_width, local_r, local_l);


    /* find the best match */
    for(d = min_d_r; d < max_d_r; d++) {
        for(i = 0; i < aggr_dim; i++) {
            for(j = 0; j < aggr_dim; j++) {
                current_right_sum += abs(local_r[((lid.y+i) * local_width) + lid.x + j + (MAX_DISP/mip_lvl)] -
                                         local_l[((lid.y+i) * local_width) + lid.x + j + d]);
            }
        }
        if(current_right_sum < best_right_sum) {
            best_right_sum = current_right_sum;
            best_right_disparity = d;
        }
        current_right_sum = 0;

    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(gid.x < (WIDTH/mip_lvl) && gid.y < (HEIGHT/mip_lvl)) {
        write_imageui(output_l, gid, best_left_disparity);
        write_imageui(output_r, gid, best_right_disparity);
    }
}

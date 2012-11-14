/*
 * Block matching stereo matching algorithm
 */
__kernel void calculateDisparity_no_def(const __global uchar *img_l, 
                                 const __global uchar *img_r,
                                 __global uchar *result,
                                 const int width,
                                 const int aggr_dim,
                                 const int aggr_radius,
                                 const int max_disp) {

    const int x = get_global_id(0), y = get_global_id(1);
    const int offset_x = x - aggr_radius;
    const int offset_y = y - aggr_radius;


    if(offset_x >= 0 && offset_x + aggr_dim < width) {
        unsigned int sum = 0, best_sum = -1, best_d = 0;
        for(int d = 0; d < max_disp; d++) {
            for(int i = offset_y; i < aggr_dim + offset_y; i++) {
                for(int j = offset_x; j < aggr_dim + offset_x; j++) {
                    sum += abs((int)img_l[i * width + j] - (int)img_r[i * width + j - d]);
                }
            }
            if(sum < best_sum) {
                best_sum = sum;
                best_d = d;
            }
            sum = 0;
        }
        result[y * width + x] = best_d;
    }
}

__kernel void calculateDisparity(const __global uchar *img_l, 
                                        const __global uchar *img_r,
                                        __global uchar *result
                                        ) {

    const int2 gid = (int2)(get_global_id(0), get_global_id(1));
    const int offset_x = gid.x - AGGR_RADIUS;
    const int offset_y = gid.y - AGGR_RADIUS;

    if(offset_x >= 0 && offset_x + AGGR_DIM < WIDTH) {
        unsigned int sum = 0, best_sum = -1, best_d = 0;
        for(int d = 0; d < MAX_DISP; d++) {
            for(int i = 0; i < AGGR_DIM; i++) {
                int y = offset_y + i;
                for(int j = 0; j < AGGR_DIM; j++) {
                    int x = offset_x + j;
                    sum += abs((int)img_l[y * WIDTH + x] - (int)img_r[y * WIDTH + x - d]);
                }
            }
            if(sum < best_sum) {
                best_sum = sum;
                best_d = d;
            }
            sum = 0;
        }
        result[gid.y * WIDTH + gid.x] = best_d;
    }
}
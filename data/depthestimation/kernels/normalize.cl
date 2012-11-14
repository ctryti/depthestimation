__kernel void normalizeResults(__global uchar *input,
                               __global uchar *output) {

    const int2 gid;
    const int2 lid;
    gid.x = get_global_id(0);
    gid.y = get_global_id(1);
    lid.x = get_local_id(0);
    lid.y = get_local_id(1);

    if(gid.x < WIDTH) {
        output[gid.y * WIDTH + gid.x] = input[gid.y * WIDTH + gid.x] * 8; ///(float)MAX_DISP)*255;

        //left[gid.y * WIDTH + gid.x] = (left[gid.y * WIDTH + gid.x]/(float)MAX_DISP)*255;
    }
}

__constant const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void normalizeResults_image2d(__read_only image2d_t input,
                                       __write_only image2d_t output,
                                       const float range) {

    const int2 gid;
    gid.x = get_global_id(0);
    gid.y = get_global_id(1);

    if(gid.x < WIDTH) {
        uchar normalized = (uchar) read_imageui(input, sampler, gid).x;
        normalized *=  8; //(normalized / range) * 255;
        write_imageui(output, gid, normalized);
    }
}

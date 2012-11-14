__constant const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

float d_calc(int2 xl,
	 	int2 xr,
		__read_only image2d_t l,
	 	__read_only image2d_t r) {

	uchar pixel_l = read_imageui(l, sampler, xl).x;

	uchar3 pixels_r;
   	pixels_r.x = read_imageui(r, sampler, (int2)(xr.x - 1, xr.y)).x;
   	pixels_r.y = read_imageui(r, sampler, (int2)(xr.x, xr.y)).x;
	pixels_r.z = read_imageui(r, sampler, (int2)(xr.x + 1, xr.y)).x;	

   	float middle_negative = (pixels_r.y + pixels_r.x) / 2.0;
   	float middle_positive = (pixels_r.y + pixels_r.z) / 2.0;
   	
   	float i_max = fmax(middle_negative, fmax(middle_positive, pixels_r.y));
   	float i_min = fmin(middle_negative, fmin(middle_positive, pixels_r.y));

	return fmax(0.0, fmax(pixel_l - i_max, i_min - pixel_l));

}

__kernel void birchfieldTomasi(__read_only image2d_t input_l,
	                       		__read_only image2d_t input_r,
                    			__write_only image2d_t output_l,
                       			__write_only image2d_t output_r
                       		   	) {
	
	const int2 gid = (int2)(get_global_id(0),get_global_id(1));
    
    float best_cost = 1000000;
    int best_d = 0;
    
    //uchar pixel_l = read_imageui(input_l, sampler, gid).x;
    
    int aggr_radius = AGGR_DIM/2;
    int offset_x = gid.x - aggr_radius;
    int offset_y = gid.y - aggr_radius;
  	float cost = 0;

    //uchar pixel_l = read_imageui(input_l, sampler, gid).x;

	for(int d = 0; d < MAX_DISP; d++) {
		for(int i = offset_y; i < offset_y + AGGR_DIM; i++) {
	    	for(int j = offset_x; j < offset_x + AGGR_DIM; j++) {
  	 			cost += fmin(d_calc((int2)(j,i), (int2)(j-d,i), input_l, input_r), 
	 						 d_calc((int2)(j-d,i), (int2)(j,i), input_r, input_l));
	 		}
	 	}
	 	if(cost < best_cost) {
	 		best_cost = cost;
	 		best_d = d;
	 	}
	 	cost = 0;
    }

    if(gid.x < WIDTH && gid.y < HEIGHT)
    	write_imageui(output_l, gid, (uchar)best_d);
}
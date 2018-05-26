__kernel void std_deviation(__local float* local_sum, __global const float* input, __global float* group_sum, float mean) {

    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    int group_id = get_group_id(0);
    int group_size = get_local_size(0);

    local_sum[local_id] = input[global_id];

    int offset;
    for (offset = group_size/2; offset > 0; offset /= 2) {
    	// wait for all local mem to get to this point
    	// and have their local_sum[local_id] available
    	// add current element + offset element
    	barrier(CLK_LOCAL_MEM_FENCE);

    	// calculate variance
    	if (local_id < offset) {
    		//printf("local sum: %f     offset: %d      localsumthing: %f\n", local_sum[local_id], offset, local_sum[local_id + offset]);
    		//local_sum[local_id] += (local_sum[local_id + offset] - mean) * (local_sum[local_id + offset] - mean);
    		local_sum[local_id] += local_sum[local_id + offset];
    		//printf("square: %f\n", (local_sum[local_id + offset]-1) * (local_sum[local_id + offset]-1));
    	}
    }

    // root of reduction sub-tree
    if (local_id == 0) {
    	group_sum[group_id] = local_sum[0];
    }
}

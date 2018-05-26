__kernel void std_deviation(__global const double* input, __global double* output, __local double* local_sum, double mean) {

    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    int group_id = get_group_id(0);
    int group_size = get_local_size(0);

   /*local_sum[local_id] = input[global_id];

    printf("global id:%lf\n", global_id);

    int offset;
    for (offset = group_size/2; offset > 0; offset /= 2) {
    	// wait for all local mem to get to this point
    	// and have their local_sum[local_id] available
    	// add current element + offset element
    	barrier(CLK_LOCAL_MEM_FENCE);

    	if (local_id < offset) {
    		local_sum[local_id] += local_sum[local_id + offset];
    		printf("input: %lf  output: %lf\n", local_sum[local_id], output[local_id]);
    	}
    }

    // root of reduction sub-tree
    if (local_id == 0) {
    	output[group_id] = local_sum[0];
    }*/

    local_sum[local_id] = input[global_id];

    printf("global id:%d  local id: %d\n", global_id, local_id);

    output[local_id] = (local_sum[local_id] - mean) * (local_sum[local_id] - mean);
    printf("input: %lf  output: %lf\n", local_sum[local_id], output[local_id]);
}

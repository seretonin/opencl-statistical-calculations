__kernel void std_deviation(__global const double* input, __global double* output, __local double* local_sum, double mean) {

    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    //printf("global id: %d\n", global_id);

    // calculate variance
    output[global_id] = (input[global_id] - mean) * (input[global_id] - mean);
    //printf("input: %lf  output: %lf\n", input[global_id], output[local_id]);
}

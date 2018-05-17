__kernel void vector_add(__global int *A, __global int *B, __global int *C) {
    
    // Get the index of the current element
    int i = get_global_id(0);

    // Do the operation
    C[i] = A[i] + B[i];
}

__kernel void maxping(__global __read_only float * a, __global __write_only float *b){
	int threadId=get_global_id(0);
	int localThreadId=get_local_id(0);
	int localSize=get_local_size(0);
	__local float fastMem[256];
	fastMem[localThreadId]=a[threadId];
	barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);

	for(int i=localSize/2;i>=1;i/=2)
	{
	    if(localThreadId<i)
	    {
	        if(fastMem[localThreadId]<fastMem[localThreadId+i])
	            fastMem[localThreadId]=fastMem[localThreadId+i];
	    }
	    barrier(CLK_LOCAL_MEM_FENCE);
	}
	if(localThreadId==0)
	    b[threadId]=fastMem[localThreadId];
}
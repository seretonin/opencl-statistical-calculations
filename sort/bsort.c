#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#define GROUP_SIZE 1024

//IF FILE IS NOT IN DIRECTORY, SEG FAULT
#define FILE_NAME "dataset_50K.txt"

#define MAX_SOURCE_SIZE (0x10000000)
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include "util.h"

double input_seq[MAX_SOURCE_SIZE];
const cl_uint ascending_order = 1;
int up = 1;
int down = 0;

/*
 * Swaps two values by comparing them in the order of the sorting
 * direction
 */
void compareSwap(double *data, int i, int j, int dir)
{

	double a,b,temp = 0.0;
	a = data[i];
	b = data[j];
	if (dir == (a > b))
	{
		temp = data[i];
		data[i] = data[j];
		data[j] = temp;
	}

}

/* 
 * Merges the bitonic sequences by comparing their respective elements
 * according to their indexes
 */
void bitonicMerge(double *data, int low, int c, int dir)
{
	int k, i;

	if (c > 1)
	{
		k = c/2;
		for (i = low ; i<low+k ;i++) {
			compareSwap(data, i, i+k, dir);  
		}
		bitonicMerge(data,low, k, dir);
		bitonicMerge(data,low+k, k, dir);  
	}
}

/*
 * Generates bitonic sequence by sorting recursively
 * two halves of the array in opposite sorting orders
 * bitonicMerge will merge the resultant data
 */
void recursiveBitonic(double *data, int low, int c, int dir)
{
	int k;
	if (c > 1)
	{
		k = c/2;
		recursiveBitonic(data,low,k,up);
		recursiveBitonic(data,low+k,k,down);
		bitonicMerge(data,low,c,dir);
	}
}

/* 
 * Sorts the entire array
 */
void sortArray(double *data,int length)
{
	recursiveBitonic(data,0,length,up);
}

/* 
 * Checks if the whole array has been sorted in the order 
 * (ascending/descending) specified
 */
int checkResult(double* data, int length, int ascend){
	
	double a,b;
    if (ascend == TRUE) {
        for(int i = 0; i < length-1; i++){
			a = data[i];
			b = data[i+1];
            if (a > b) {
                return FALSE;
            }
        }
    } 
	else {
        for(int i = 0; i < length; i++){
            if(data[i] < data[i+1]){
                return FALSE;
            }
        }
    }
    return TRUE;
}

/* 
 * Counts the number of data in the specified file
 */
int countDataEntries()
{
	FILE *file = fopen(FILE_NAME, "r");
	if (file == NULL) {
		//SEG FAULT
		printf("file is not in the directory\n");
	}
	int count = 0;

	while(!feof(file))
	{
	  char ch = fgetc(file);
	  if(ch == '\n')
	  {
		count++;
	  }
	}
	return count;
}

/* 
 * Reads and stores the data from the file in the array specified and
 * up to the length (dataset size) specified 
 */
void storeDataToProcess(double *data,int length)
{

	FILE *file = fopen(FILE_NAME, "r");
	double num;
	int i = 0;		
	while(fscanf(file, "%lf" ,&num) > 0)
	{
		if (i >= length) {
			break;
		}
		data[i] = num;	
		i++;
	}
	fclose(file);
}

/* 
 * Obtains the current time as of when the function is being called
 */
double getTime(){
	
  struct timeval t;
  double sec, msec;
  
  while (gettimeofday(&t, NULL) != 0);
  sec = t.tv_sec;
  msec = t.tv_usec;
  
  sec = sec + msec/1000000.0;
  
  return sec;
}

int main(int argc, char *argv[])
{
	double t1 = 0.0;
	double t2 = 0.0; 
	double t_seq = 0.0, t_par = 0.0;
	double median = 0.0, min = 0.0, max = 0.0;	
	double low_q = 0.0, upper_q = 0.0;

	//input and output array on host
    double* input = NULL; 
    double* output = NULL; 
    	
	int length = 0;
	int dataset_count = 0;
	double div = 0.0;
	dataset_count = countDataEntries();
	div = log(dataset_count) / log(2);
	length = pow(2,(int)div);
	if (length != dataset_count) {
		printf("dataset has been resized to %d (2^%d) bcs it is not divisible by work group size\n",length,(int)div);
	}	
	
	int datasize = 0;
	datasize = length*sizeof(double);
	
	input = (double *)malloc(length *sizeof(double));
	if (input == NULL) {
		printf("memory allocation for input was not succesful\n");
	}
	
	storeDataToProcess(input,length);
	storeDataToProcess(input_seq,length);

	int mid = (length)/2;
	int mid_1 = mid - 1;
	int low_odd = 0, low_even = 0, low_even_1 = 0;
	int upp_odd = 0, upp_even =0 , upp_even_1 =0;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//START OF SEQUENTIAL CALCULATIONS////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	printf("sequential sorting in progress..\n");
	t1 = getTime();
	sortArray(input_seq,length);
	//calculation for median and quartiles takes into account of the length of the data (odd/even)
	if (mid % 2 == 0) {
		low_even = (mid)/2;
		low_even_1 = low_even - 1;
		upp_even = length - low_even - 1;
		upp_even_1 = length - low_even; 
		low_q = (input_seq[low_even] + input_seq[low_even_1]) / 2;
		upper_q = (input_seq[upp_even] + input_seq[upp_even_1]) / 2;
	}
	else {
		low_odd = (int)(length/4);
		upp_odd = (int)((3*length)/4);
		low_q = input_seq[low_odd];
		upper_q = input_seq[upp_odd];
	}		
	median = (input_seq[mid] + input_seq[mid_1]) / 2;
	min = input_seq[0];
	max = input_seq[length-1];	
	t2 = getTime();
	t_seq = t2 - t1;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//END OF SEQUENTIAL CALCULATIONS//////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	
	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//START OF OPENCL PARALLEL CALCULATIONS///////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	printf("parallel sorting in progress..\n");
	cl_int error = 0;
	
	//configure the work-item structure
	size_t globalThreads = length;    
    size_t threadsPerGroup = GROUP_SIZE;
    	
    output = (double *)malloc(length * sizeof(double));
    if (output == NULL) {
		printf("memory allocation for output failed");
	}
 
 	memset(input, 0, length * sizeof(double));
	memset(output, 0, length * sizeof(double));
   
	int k = 0;
	for(k = 0; k < length; k++)
	{
		output[k] = 0.0;
	}
  
	cl_device_id devices = NULL;
	error = clGetDeviceIDs(NULL,CL_DEVICE_TYPE_GPU,1,&devices,NULL);

    cl_uint numPlatforms = 0;
	cl_platform_id *platforms = NULL;
	error = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (error == CL_SUCCESS) {
	    assert(error == CL_SUCCESS);
    }
	platforms =	(cl_platform_id*)malloc(numPlatforms*sizeof(cl_platform_id));

	//create a context
	cl_context ctx = NULL;
	ctx = clCreateContext(NULL, 1, &devices, NULL, NULL, &error);

	//create a command queue
	cl_command_queue queue;
	queue = clCreateCommandQueue(ctx, devices, 0, &error);
	
	//create a program
	FILE *fp;
	char *pgmSource = (char *)readFile("bsortKernel.cl");
	size_t source_size;

	fp = fopen("bsortKernel.cl", "r");
	if (!fp) {
	  printf("Failed to load kernel file.\n");
	  exit(1);
	}
	pgmSource = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(pgmSource, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);
	
	
	cl_program program = clCreateProgramWithSource(ctx,1,(const char**)&pgmSource,(const size_t*)&source_size,&error);
	
 	//build (compile) the program for the device(s)
	error = clBuildProgram(program,1, &devices, NULL, NULL, NULL);
     
	//create the kernel
	cl_kernel kernel = NULL;
	kernel = clCreateKernel(program, "parallelBitonicSort", &error);
	   
 	//create device buffers
    cl_mem inputBuffer = NULL;  //input array on the device
	inputBuffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE,datasize, NULL, NULL);
	if (inputBuffer == NULL)
	{
		printf("creation of input buffer failed\n");
	}

	cl_mem outputBuffer = NULL;
	outputBuffer = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,datasize, NULL, NULL);
   
 	//write host data to device buffers
	error = clEnqueueWriteBuffer(queue, inputBuffer, CL_FALSE, 0,datasize, input, 0, NULL, NULL);

	error = clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0,datasize, output, 0, NULL, NULL);
    
 	//set the kernel arguments
 	error = 0;
    error = clSetKernelArg(kernel, 0, sizeof(cl_mem),(void*)&inputBuffer);
    error |= clSetKernelArg(kernel, 3, sizeof(cl_uint),(void*)&ascending_order);
   
	t1 = getTime();
	
	//enqueue the kernel for execution
    cl_uint stages = 0;
    for(unsigned int i = length; i > 1; i >>= 1){
        ++stages;
    }

	for(cl_uint stage = 0; stage < stages; ++stage) {
        clSetKernelArg(kernel, 1, sizeof(cl_uint),(void*)&stage);
        for(cl_uint subStage = 0; subStage < stage +1; subStage++) {
            clSetKernelArg(kernel, 2, sizeof(cl_uint),(void*)&subStage);
            cl_event exeEvt;
            error = clEnqueueNDRangeKernel(queue,kernel,1,NULL,&globalThreads,&threadsPerGroup,0,NULL,&exeEvt);
            clWaitForEvents(1, &exeEvt);
		}
    }    
    	
	//read the inputBuffer content back to the host (output)
	error = clEnqueueReadBuffer(queue,inputBuffer,CL_TRUE,0,length*sizeof(double),output,0,NULL,NULL);
	
	//calculate the statistical values
	
	//calculation for median and quartiles takes into account of the length of the data (odd/even)
	if (mid % 2 == 0) {
		low_even = (mid)/2;
		low_even_1 = low_even - 1;
		upp_even = length - low_even - 1;
		upp_even_1 = length - low_even; 
		low_q = (input_seq[low_even] + input_seq[low_even_1]) / 2;
		upper_q = (input_seq[upp_even] + input_seq[upp_even_1]) / 2;
	}
	else {
		low_odd = (int)(length/4);
		upp_odd = (int)((3*length)/4);
		low_q = input_seq[low_odd];
		upper_q = input_seq[upp_odd];
	}		
	median = (input_seq[mid] + input_seq[mid_1]) / 2;
	min = input_seq[0];
	max = input_seq[length-1];	
	
	t2 = getTime();
	t_par = t2 - t1;
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//END OF OPENCL PARALLEL CALCULATIONS/////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////

	
	//to suppress the printing of sorted data, suppress = 1
	//to print out the sorted data, suppress = 0
	int suppress = 1;
	
	if (suppress == 0) {	
		printf("\n-AFTER SORTING------------------------\n");	
		for(unsigned int i = 0; i < length; i++) {
			printf("%lf(%i)\n", input_seq[i],i); 
		}
	}

	//check if output data from host and sequential array has been sorted
	int rot = 0;
	rot = checkResult(input_seq, length, ascending_order);
	if (rot == TRUE) {
			printf("data has been sorted\n");
	}
    int ret = 0;
	ret = checkResult(output, length, ascending_order);
	printf("\n");
	printf("-PROGRAM DETAILS------------------------");
	printf("\n\n");		
	printf("dataset         : %s\n",FILE_NAME);
	printf("number of data  : %d\n",length);   
	printf("work group size : %d\n",GROUP_SIZE);

	printf("\n");
	printf("-SEQUENTIAL----------------------------");
	printf("\n\n");		
	printf("min             : %lf\n",min);
	printf("lower quartile  : %lf\n",low_q);
	printf("median          : %lf\n",median);
	printf("upper quartile  : %lf\n",upper_q);
	printf("max             : %lf\n",max);
	printf("time taken      : %6.6f secs\n",t_seq);			
	
    if (ret == TRUE) {
		printf("\n");		
		printf("-PARALLEL-------------------------------");
		printf("\n\n");				
		printf("min             : %lf\n",min);
		printf("lower quartile  : %lf\n",low_q);
		printf("median          : %lf\n",median);
		printf("upper quartile  : %lf\n",upper_q);
		printf("max             : %lf\n",max);
		printf("time taken      : %6.6f secs\n",t_par);	
		printf("speedup         : %6.6f\n\n",t_seq/t_par);					
	}							
    else {
		printf("\n");			
		printf("----------------------------------------");
		printf("\n\n");				
        printf("results are unavailable. data is NOT sorted");
		printf("\n\n");	  
    }

	//free OpenCL resources
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);

	clReleaseMemObject(inputBuffer);

	clReleaseContext(ctx);

	//free host resources
	Free(platforms);
	Free(devices);
	Free(pgmSource);
    Free(input);
    Free(output);

	
    return 0;
}

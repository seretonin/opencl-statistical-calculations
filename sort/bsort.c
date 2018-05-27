#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>
#include <time.h>
#define GROUP_SIZE 64
#define FILE_NAME "dataset_1M.txt"

#define MAX_SOURCE_SIZE (0x10000000)
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include "util.h"

double input_seq[MAX_SOURCE_SIZE];
const cl_uint ascending_order = 1;

void compare_swap(double *data, int i, int j, int dir)
{

	double a,b,temp = 0.0;
	a = data[i];
	b = data[j];
	if (a > b)
	{
		temp = data[i];
		data[i] = data[j];
		data[j] = temp;
	}
	else {
		//printf("already sorted: %f and %f\n",a,b);
	}
}

void bitonicmerge(double *data, int low, int c, int dir)
{
	int k, i;

	if (c > 1)
	{
		k = c/2;
		for (i = low;i < low+k ;i++)
			compare_swap(data, i, i+k, dir);    
			bitonicmerge(data,low, k, dir);
			bitonicmerge(data,low+k, k, dir);    
	}
}

/*
 * Generates bitonic sequence by sorting recursively
 * two halves of the array in opposite sorting orders
 * bitonicmerge will merge the resultant data
 */
void recbitonic(double *data, int low, int c, int dir)
{
	int k;
	if (c > 1)
	{
		k = c/2;
		recbitonic(data,low,k,1);
		recbitonic(data,low+k,k,0);
		bitonicmerge(data,low,c,dir);
	}
}

/* 
 * Sorts the entire array
 */
void sort(double *data,int length)
{
	recbitonic(data,0,length,1);
	//printf("length: %d\n",length);
}

int checkResult(double* data, int length, int ascend){
	
	double a,b;
    if (ascend == TRUE) {
        for(int i = 0; i < length-1; i++){
			a = data[i];
			b = data[i+1];
			//printf("compare: %f and %f index: %d\n",a,b,i);
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

int countDataEntries()
{
	FILE *file = fopen(FILE_NAME, "r");
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

void storeDataToProcess(double *data)
{

	FILE *file = fopen(FILE_NAME, "r");
	double num;
	int i = 0;
	//printf("\n-INPUT DATA----------------------------\n");		
	while(fscanf(file, "%lf" ,&num) > 0)
	{
		data[i] = num;
		//printf("%f\n", data[i]); 		
		i++;
	}
	fclose(file);
}

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
	double tw1, tw2 = 0.0;
	double t_seq, t_par = 0.0;
	double median, min, max = 0.0;	
	//double median_par, min_par, max_par = 0.0;	

	tw1 = getTime();
    double* input;    //input array on host
    double* output;   //output array on host
    	
	int length;
	length = countDataEntries();
	
	int datasize;
	datasize = length*sizeof(double);
	
	input = (double *)malloc(length *sizeof(double));
	if (input == NULL) {
		printf("memory allocation for input was not succesful");
	}
	
	storeDataToProcess(input);
	storeDataToProcess(input_seq);
	

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//START OF SEQUENTIAL CALCULATIONS////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	t1 = getTime();
	int mid = (length-1)/2;
	sort(input_seq,length);
	median = input_seq[mid];
	min = input_seq[1];
	max = input_seq[length-2];	
	t2 = getTime();
	t_seq = t2 - t1;

	//print the input data
	printf("\n-AFTER SORTING------------------------\n");	
	/*for(unsigned int i = 0; i < length; i++) {
		printf("%f\n", input_seq[i]); 
	}*/
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//END OF SEQUENTIAL CALCULATIONS//////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	
	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//START OF OPENCL PARALLEL CALCULATIONS///////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////

	cl_int error;
	
	//configure the work-item structure
	size_t globalThreads = length;    
    size_t threadsPerGroup = GROUP_SIZE;
    
    //int num_of_work_groups = length/threadsPerGroup;
    	
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
	//clCheckEqWithMsg(error, CL_SUCCESS, "creation of context was not successful");

	//create a command queue
	cl_command_queue queue;
	queue = clCreateCommandQueue(ctx, devices, 0, &error);
	//clCheckEqWithMsg(error, CL_SUCCESS, "creation of command queue was not succesful");
	
	//create a program using clCreateProgramWithSource()
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
    //clCheckEqWithMsg(error, CL_SUCCESS, "can't create program..");
	
 	//build (compile) the program for the device(s)
	error = clBuildProgram(program,1, &devices, NULL, NULL, NULL);
    //clCheckEqWithMsg(error, CL_SUCCESS, "Can't build program.");
     
	//create the kernel
	cl_kernel kernel = NULL;
	kernel = clCreateKernel(program, "parallelBitonicSort", &error);
	//clCheckEqWithMsg(error, CL_SUCCESS, "Can't get kernel from program.");
	   
 	//create device buffers
    cl_mem inputBuffer = NULL;  //input array on the device
	inputBuffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE,datasize, NULL, NULL);
	if (inputBuffer == NULL)
	{
		printf("something wrong with input buffer??");
	}
    //clCheckEqWithMsg(error, CL_SUCCESS, "can't create device buffer...\n");

	cl_mem outputBuffer = NULL;
	outputBuffer = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,datasize, NULL, NULL);
    //clCheckEqWithMsg(error, CL_SUCCESS, "can't create device buffer...\n");
   
 	//write host data to device buffers
	error = clEnqueueWriteBuffer(queue, inputBuffer, CL_FALSE, 0,datasize, input, 0, NULL, NULL);
    //clCheckEqWithMsg(error, CL_SUCCESS, "can't write host data to device buffer.\n");

	error = clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0,datasize, output, 0, NULL, NULL);
    //clCheckEqWithMsg(error, CL_SUCCESS, "can't write host data to device buffer.\n");   
    
 	//set the kernel arguments
 	error = 0;
    error = clSetKernelArg(kernel, 0, sizeof(cl_mem),(void*)&inputBuffer);
    error |= clSetKernelArg(kernel, 3, sizeof(cl_uint),(void*)&ascending_order);
	//clCheckEqWithMsg(error, CL_SUCCESS, "Can't set kernel's arguments.");
   
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
            //clCheckEqWithMsg(error, CL_SUCCESS, "Kernel execution failure!\n");
		}
    }    
    	
	//read the inputBuffer content back to the host (output)
	error = clEnqueueReadBuffer(queue,inputBuffer,CL_TRUE,0,length*sizeof(double),output,0,NULL,NULL);
	//clCheckEqWithMsg(error, CL_SUCCESS, "can't write to host from device.\n");
	
	//calculate the statistical values
	median = input_seq[mid];
	min = input_seq[1];
	max = input_seq[length-1];
	
	t2 = getTime();
	t_par = t2 - t1;
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//END OF OPENCL PARALLEL CALCULATIONS/////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//to suppress the printing of output data
	int suppress = 1;
	
	if (suppress == 0) {
		//print the sorted data
		for(unsigned int i = 0; i < length; i++) {
			printf("%lf\t\t\t\t", input_seq[i]); 
			if (i%3 == 0) {
				printf("\n");
			}
		}
	}
 	tw2 = getTime();
	printf("\n");
	printf("-PROGRAM DETAILS------------------------");
	printf("\n\n");		
	printf("dataset         : %s\n",FILE_NAME);
	printf("number of data  : %d\n",length);   
	printf("work group size : %d\n",GROUP_SIZE);

	printf("\n");
	printf("-SEQUENTIAL----------------------------");
	printf("\n\n");		
	printf("median          : %lf\n",median);
	printf("min             : %lf\n",min);
	printf("max             : %lf\n",max);
	printf("time taken      : %6.6f secs\n",t_seq);			
	
	//check if output data has been sorted
    int ret = checkResult(output, length, ascending_order);
    if (ret == TRUE) {
		printf("\n");		
		printf("-PARALLEL-------------------------------");
		printf("\n\n");		
		printf("median          : %lf\n",median);
		printf("min             : %lf\n",min);
		printf("max             : %lf\n",max);
		printf("time taken      : %6.6f secs\n",t_par);				
	}							
    else {
		printf("\n");			
		printf("----------------------------------------");
		printf("\n\n");				
        printf("results are unavailable. data is NOT sorted");
		printf("\n\n");	  
    }

	printf("\n");		
	printf("whole program took %6.6f secs\n",tw2-tw1);		
	printf("----------------------------------------");	
	printf("\n");	

	//release OpenCL resources

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


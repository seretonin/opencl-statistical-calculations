#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>
#include <time.h>

#include <fcntl.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
//to suppress warnings on deprecated APIs
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define GROUP_SIZE 256
#define FILE_NAME "integer.csv"
#include <CL/cl.h>
#include "util.h"
#include "clutil.h"

//descending order = 0
//ascending order = 1
const cl_uint sort_order = 1;
//load data from file = 1
//generate random data = 0 
int load_file = 1;
unsigned int INPUT_LENGTH = 1<<16;

int checkResult(int *data, int length, int ascend){
    if (ascend == TRUE){
        for(int i = 0; i < length-1; i++){
            if(data[i] > data[i+1]){
                return FALSE;
            }
        }
    }else{
        for(int i = 0; i < length; i++){
            if(data[i] < data[i+1]){
                return FALSE;
            }
        }
    }
    return TRUE;
}

void fillRandomData(int *data, int length){
    srand((unsigned)time(NULL));
    for(int i = 0; i < length; i++){
        data[i] = rand();
    }
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
	
	printf("# of elements: %d\n",count);
	return count;
}

void storeDataToProcess(int* data)
{
	printf("here_1\n");
	FILE *file = fopen(FILE_NAME, "r");
	int i = 0;
	int c;
	
	if (file == NULL) {
		printf("file failed to load\n");
		exit(1);
	}
	else {
		while(1) {
			c = fgetc(file);
			if (feof(file)) {
				break;
			}
		printf("%c",c);
		//data[i] = atoi(c);
		}
	} 
	
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
    cl_int err;
    cl_uint numPlatforms = 0;
	cl_platform_id *platforms = NULL;

    int length;
    int *input = NULL;    //input array on host
    int *output = NULL;   //output array on host
    cl_mem inputBuffer = NULL;  //input array on the device

	input = (int *)malloc(INPUT_LENGTH * sizeof(int));
	output = (int *)malloc(INPUT_LENGTH * sizeof(int));

	memset(input, 0, INPUT_LENGTH * sizeof(int));
	memset(output, 0, INPUT_LENGTH * sizeof(int));
	
	if (load_file == 0) {
		length = INPUT_LENGTH;
		fillRandomData(input, length);
	}
	else {
		length = countDataEntries();
		storeDataToProcess(input);
		
	}
	const int datasize = length * sizeof(int);
	//retrieve # of platforms
	err = clGetPlatformIDs(0, NULL, &numPlatforms);
    assert(err == CL_SUCCESS);

	    if (numPlatforms > 0) {
        assert(numPlatforms > 0);
    }
    printf("found %d platforms.\n", numPlatforms);

	//memory allocation on platform(s)
	platforms =	(cl_platform_id*)malloc(numPlatforms*sizeof(cl_platform_id));
    if (platforms != NULL) {	
        assert(platforms != NULL);
    }

    //fill in platform(s)
	err = clGetPlatformIDs(numPlatforms, platforms, NULL);
    if (err == CL_SUCCESS) {
	    assert(err == CL_SUCCESS);
    }

    cl_uint numDevices = 0;
	cl_device_id *devices = NULL;

	//retrieve # of GPU(s)
	err = clGetDeviceIDs(platforms[0],
                         CL_DEVICE_TYPE_GPU, 
                         0,
                         NULL,
                         &numDevices);
    if(numDevices > 0){
        printf("found %d GPUs for platform 0 \n", numDevices);
    }else{
        err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU,
                                0, NULL, &numDevices);
        if(numDevices > 0){
            printf("found %d CPUs for platform 0 \n", numDevices);
        }else{
            printf("can't find CPU or GPU devices\n");
            exit(-1);
        }
    }

    if (err == CL_SUCCESS) {
	    assert(err == CL_SUCCESS);
    }

	//memory allocation on device(s)
	devices = (cl_device_id*)malloc(numDevices*sizeof(cl_device_id));
    if (devices != NULL) {
        assert(devices);
    }	
    

    err = clGetDeviceIDs(
		platforms[0],
		CL_DEVICE_TYPE_GPU,
		numDevices,
		devices,
		NULL);
	clCheckEqWithMsg(err, CL_SUCCESS, "can't get devices.");


	//create a context
	cl_context ctx = NULL;

	ctx = clCreateContext(NULL, numDevices, devices, NULL, NULL, &err);
	clCheckEqWithMsg(err, CL_SUCCESS, "Can't create context.");

	//create a command queue
	cl_command_queue queue;

	queue = clCreateCommandQueue(ctx, devices[0], 0, &err);
	clCheckEqWithMsg(err, CL_SUCCESS, "Can't create command queue..");

	//create device buffers
	inputBuffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                 datasize, NULL, &err);
    clCheckEqWithMsg(err, CL_SUCCESS, "can't create device buffer...\n");

	//write host data to device buffers
	err = clEnqueueWriteBuffer(queue, inputBuffer, CL_FALSE, 0,
                               datasize, input, 0, NULL, NULL);
    clCheckEqWithMsg(err, CL_SUCCESS, "can't write host data to device buffer.\n");


	//create and compile the program
	char *pgmSource = (char *)readFile("bsortKernel.cl");

	//create a program using clCreateProgramWithSource()
	cl_program program = clCreateProgramWithSource(ctx,
                                                   1,
                                                   (const char**)&pgmSource,
                                                   NULL,
                                                   &err);
    clCheckEqWithMsg(err, CL_SUCCESS, "can't create program..");
    
	//build (compile) the program for the device(s)
	err = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);
    clCheckEqWithMsg(err, CL_SUCCESS, "Can't build program.");
    
	//create the kernel
	cl_kernel kernel = NULL;

	kernel = clCreateKernel(program, "parallelBitonicSort", &err);
	clCheckEqWithMsg(err, CL_SUCCESS, "Can't get kernel from program.");
	
	//set the kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem),(void*)&inputBuffer);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_uint),(void*)&sort_order);

	clCheckEqWithMsg(err, CL_SUCCESS, "Can't set kernel's arguments.");

	//enqueue the kernel for execution
    cl_uint stages = 0;
    for(unsigned int i = INPUT_LENGTH; i > 1; i >>= 1){
        ++stages;
    }

	//configure the work-item structure
	size_t globalThreads[1] = {INPUT_LENGTH/2};
    
    size_t threadsPerGroup[1] = {GROUP_SIZE};

	t1 = getTime();
	for(cl_uint stage = 0; stage < stages; ++stage) {
        clSetKernelArg(kernel, 1, sizeof(cl_uint),(void*)&stage);

        for(cl_uint subStage = 0; subStage < stage +1; subStage++) {
            clSetKernelArg(kernel, 2, sizeof(cl_uint),(void*)&subStage);
            cl_event exeEvt;
            err = clEnqueueNDRangeKernel(queue,
										kernel,
										1,
										NULL,
										globalThreads,
										threadsPerGroup,
										0,
										NULL,
										&exeEvt);
            clWaitForEvents(1, &exeEvt);
            clCheckEqWithMsg(err, CL_SUCCESS, "Kernel execution failure!\n");

		}
    }
	
	//read the output buffer back to the host
	clEnqueueReadBuffer(queue,
                        inputBuffer,
                        CL_TRUE,
                        0,
                        datasize,
                        output,
                        0,
                        NULL,
                        NULL);

	int median = output[INPUT_LENGTH/2];
	int min = output[0];
	int max = output[INPUT_LENGTH-1];
	t2 = getTime();
	
	//print the sorted data
    for(unsigned int i = 0; i < INPUT_LENGTH; i++) {
        printf("%d\t", output[i]); 
        if (i%5 == 0) {
			printf("\n");
		} 
	}
	
    int ret = checkResult(output, INPUT_LENGTH, sort_order);
    if (ret == TRUE) {
		printf("\n-----------------------------------------\n");
		printf("INPUT_LENGTH : %d\n",INPUT_LENGTH);
		printf("median       : %d\n",median);
		printf("min          : %d\n",min);
		printf("max          : %d\n",max);
		printf("time taken   : %6.5f secs\n",(t2 - t1));		
		printf("-------------------------------------------\n");		
	}							
    else {
		printf("\n-----------------------------------------\n");		
        printf("results are unavailable. data is NOT sorted\n");
		printf("\n-----------------------------------------\n");    
    }
    
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
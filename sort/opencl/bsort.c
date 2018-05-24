#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>
#include <time.h>

//to suppress warnings on deprecated APIs
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include "util.h"
#include "clutil.h"
//how big should the group size be pls help
#define GROUP_SIZE 32
const unsigned int INPUT_ARRAY_LENGTH = 1000000;

//descending order = 0
//ascending order = 1
const cl_uint sorting_order = 1; 

double getTime(){
  struct timeval t;
  double sec, msec;
  
  while (gettimeofday(&t, NULL) != 0);
  sec = t.tv_sec;
  msec = t.tv_usec;
  
  sec = sec + msec/1000000.0;
  
  return sec;
}

/* This function checks if the array has been sorted 
 * ascendingly/descendingly, depending on the parameter ascend and 
 * returns a TRUE/FALSE */
int checkIfArrayIsSorted(int *data, int array_length, int ascend) {
	int i,j;
	if (ascend) {
		for (i = 0; i < array_length - 1; i++) {
			if (data[i] > data[i+1]) {
				return FALSE;
			}
		}
	}
	else {
		if (ascend) {
			for (j = 0; j < array_length - 1; j++) {
				if (data[i+1] > data[i+1]) {
					return FALSE;
				}
			}	
		}
	}
	return TRUE;
}

/* This function populates the array with random numbers generated from 
 * the current time */
void populateArray(int *data, int array_length){
    srand((unsigned)time(NULL));
    for(int i = 0; i < array_length; i++){
        data[i] = rand();
    }
}

/* This function prints the input/output array passed in */
void printArray(int *array) {
	for(unsigned int i = 0; i < INPUT_ARRAY_LENGTH; i++) {
        printf("%d\n", array[i]); 
	}
}

int main(int argc, char *argv[])
{
	double t1 = 0.0;
	double t2 = 0.0; 		
    cl_int error_code;
    cl_uint num_of_platforms = 0;
	cl_platform_id *platforms = NULL;

    const int length = INPUT_ARRAY_LENGTH;
    const int datasize = length * sizeof(int);
    int *input_array = NULL;    //input array on host
    int *output_array = NULL;   //output array on host
    cl_mem inputBuffer = NULL;  //input array on the device

	input_array = (int *)malloc(INPUT_ARRAY_LENGTH * sizeof(int));
	output_array = (int *)malloc(INPUT_ARRAY_LENGTH * sizeof(int));

	memset(input_array, 0, INPUT_ARRAY_LENGTH * sizeof(int));
	memset(output_array, 0, INPUT_ARRAY_LENGTH * sizeof(int));
	populateArray(input_array, length);

	error_code = clGetPlatformIDs(0, NULL, &num_of_platforms);
	
    if (error_code == CL_SUCCESS) {
	    assert(error_code == CL_SUCCESS);\
	}
	else {
		printf("Error. error_code: %d\n", error_code);
	}
	
    if (num_of_platforms > 0) {
        assert(num_of_platforms);
        printf("Found %d platform(s)\n", num_of_platforms);
    }
    else {
		printf("No platform found\n");
	}
    

	// Allocate enough space for each platform
	platforms =	(cl_platform_id*)malloc(num_of_platforms*sizeof(cl_platform_id));
    if (platforms != NULL) {	
        assert(platforms);
    }

    // Fill in platforms with clGetPlatformIDs()
	error_code = clGetPlatformIDs(num_of_platforms, platforms, NULL);
    if (error_code == CL_SUCCESS) {
	    assert(error_code == CL_SUCCESS);
	    printf("Platforms asserted successfully\n");
    }


    cl_uint numDevices = 0;
	cl_device_id *devices = NULL;

	// Use clGetDeviceIDs() to retrieve the number of
	// devices present
	error_code = clGetDeviceIDs(platforms[0],
                         CL_DEVICE_TYPE_GPU,     //only get GPU
                         0,
                         NULL,
                         &numDevices);
    if(numDevices > 0){
        printf("Found %d GPUs for platform 0 \n", numDevices);
    }else{
        error_code = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU,
                                0, NULL, &numDevices);
        if(numDevices > 0){
            printf("Found %d GPUs for platform 0 \n", numDevices);
            printf("wtf opencl");
        }else{
            printf("can't find GPU or GPU devices\n");
            exit(-1);
        }
    }

    if (error_code == CL_SUCCESS) {
	    assert(error_code == CL_SUCCESS);
    }

	// Allocate enough space for each device
	devices = (cl_device_id*)malloc(numDevices*sizeof(cl_device_id));
    if (devices != NULL) {
        assert(devices);
    }	
    

    error_code = clGetDeviceIDs(
		platforms[0],
		CL_DEVICE_TYPE_GPU,
		numDevices,
		devices,
		NULL);
	clCheckEqWithMsg(error_code, CL_SUCCESS, "can't get devices.");

    //-----------------------------------------------------
	// Create a context
	//-----------------------------------------------------
	cl_context ctx = NULL;

	ctx = clCreateContext(NULL, numDevices, devices, NULL, NULL, &error_code);
	clCheckEqWithMsg(error_code, CL_SUCCESS, "Can't create context.");

	//-----------------------------------------------------
	// Create a command queue
	//-----------------------------------------------------

	cl_command_queue queue;

	queue = clCreateCommandQueue(ctx, devices[0], 0, &error_code);
	clCheckEqWithMsg(error_code, CL_SUCCESS, "Can't create command queue..");

	//-----------------------------------------------------
	// Create device buffers
	//-----------------------------------------------------
	inputBuffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                 datasize, NULL, &error_code);
    clCheckEqWithMsg(error_code, CL_SUCCESS, "can't create device buffer...\n");
	//-----------------------------------------------------
	// Write host data to device buffers
	//-----------------------------------------------------
	error_code = clEnqueueWriteBuffer(queue, inputBuffer, CL_FALSE, 0,
                               datasize, input_array, 0, NULL, NULL);
    clCheckEqWithMsg(error_code, CL_SUCCESS, "can't write host data to device buffer.\n");
	//-----------------------------------------------------
	//Create and compile the program
	//-----------------------------------------------------
	char *pgmSource = (char *)readFile("bsortKernel.cl");

	// Create a program using clCreateProgramWithSource()
	cl_program program = clCreateProgramWithSource(ctx,
                                                   1,
                                                   (const char**)&pgmSource,
                                                   NULL,
                                                   &error_code);
    clCheckEqWithMsg(error_code, CL_SUCCESS, "can't create program..");
	// Build (compile) the program for the devices with clBuildProgram()
	error_code = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);
    clCheckEqWithMsg(error_code, CL_SUCCESS, "Can't build program.");
	//-----------------------------------------------------
	// Create the kernel
	//-----------------------------------------------------
	cl_kernel kernel = NULL;

	kernel = clCreateKernel(program, "parallelBitonicSort", &error_code);
	clCheckEqWithMsg(error_code, CL_SUCCESS, "Can't get kernel from program.");
	//-----------------------------------------------------
	// Set the kernel arguments
	//-----------------------------------------------------
    error_code = clSetKernelArg(kernel, 0, sizeof(cl_mem),(void*)&inputBuffer);
    error_code |= clSetKernelArg(kernel, 3, sizeof(cl_uint),(void*)&sorting_order);

	clCheckEqWithMsg(error_code, CL_SUCCESS, "Can't set kernel's arguments.");

	//-----------------------------------------------------
	// Enqueue the kernel for execution
	//-----------------------------------------------------
    cl_uint stages = 0;
    for(unsigned int i = INPUT_ARRAY_LENGTH; i > 1; i >>= 1){
        ++stages;
    }

    size_t globalThreads[1] = {INPUT_ARRAY_LENGTH/2};
    size_t threadsPerGroup[1] = {GROUP_SIZE};
	
	t1 = getTime();
	for(cl_uint stage = 0; stage < stages; ++stage) {
        clSetKernelArg(kernel, 1, sizeof(cl_uint),(void*)&stage);

        for(cl_uint subStage = 0; subStage < stage +1; subStage++) {
            clSetKernelArg(kernel, 2, sizeof(cl_uint),(void*)&subStage);
            cl_event exeEvt;
            cl_ulong executionStart, executionEnd;
            error_code = clEnqueueNDRangeKernel(queue,
                                         kernel,
                                         1,
                                         NULL,
                                         globalThreads,
                                         threadsPerGroup,
                                         0,
                                         NULL,
                                         &exeEvt);
            clWaitForEvents(1, &exeEvt);
            clCheckEqWithMsg(error_code, CL_SUCCESS, "Kernel execution failure!\n");

            clGetEventProfilingInfo(exeEvt, CL_PROFILING_COMMAND_START, sizeof(executionStart), &executionStart, NULL);
            clGetEventProfilingInfo(exeEvt, CL_PROFILING_COMMAND_END, sizeof(executionEnd), &executionEnd, NULL);
            clReleaseEvent(exeEvt);
            
			//printf("\n %lu\t",executionStart);
			//printf("%lu",executionEnd);
            //printf("Execution of the bitonic sort took %lu.%lu s\n", (executionEnd - executionStart)/1000000000, (executionEnd - executionStart)%1000000000);
        }
    }
    t2 = getTime();

	// Read the output buffer back to the host
	clEnqueueReadBuffer(queue,
                        inputBuffer,
                        CL_TRUE,
                        0,
                        datasize,
                        output_array,
                        0,
                        NULL,
                        NULL);

	//print the sorted data
	//printArray(output_array);

	
	printf("INPUT_ARRAY_LENGTH: %d\n",INPUT_ARRAY_LENGTH);
	printf("GROUP_SIZE: %d\n",GROUP_SIZE);
    printf("time: %6.2f secs\n",(t2 - t1));
    int ret = checkIfArrayIsSorted(output_array, INPUT_ARRAY_LENGTH, sorting_order);
    if (ret == TRUE)
        printf("data has been sorted :)\n");
    else
        printf("data sorting failed :(\n");
        
 
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
    Free(input_array);
    Free(output_array);

    return 0;
}

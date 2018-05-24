//hi
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
//include windowsopencl
#include <OpenCL/opencl.h>

#define WORK_GROUP_SIZE 64
#define FILE_NAME "float.csv"
#define GPU "GeForce"

int data_size = 0;

const char *parallelSum_kernel = "\n" \
"__kernel void parallelSum(__local float* localSum, __global const float* input, __global float* groupSum) \n" \
"{                                                                                  \n" \
"   int localID   = get_local_id(0);                                                \n" \
"   int globalID  = get_global_id(0);                                               \n" \
"   int groupID = get_group_id(0);                                                 \n" \
"   int groupSize = get_local_size(0);                                              \n" \
"                                                                                   \n" \
"   localSum[localID] = input[globalID];                                            \n" \
"                                                                                   \n" \
"   for(int stride = groupSize / 2; stride > 0; stride /= 2)                      \n" \
"   {                                                                               \n" \
"      //wait for all local memory to get to this point and have their localSum[localID]              \n" \
"      //available. This is so that we can add the current element + stride element                   \n" \
"      barrier(CLK_LOCAL_MEM_FENCE);                                                \n" \
"      if(localID < stride)                                                         \n" \
"      {                                                                            \n" \
"        localSum[localID] += localSum[localID + stride];                        \n" \
"      }                                                                            \n" \
"   }                                                                               \n" \
"                                                                                   \n" \
"   if(localID == 0)                                                                \n" \
"   {                                                                               \n" \
"      groupSum[groupID] = localSum[0];                                             \n" \
"   }                                                                               \n" \
"}                                                                                  \n" \
"\n";

// void readGPUResult(float* results)
// {
//   int i;
//   float sumOfWorkSums = 0.0;
//   for(i = 0; i < data_size/WORK_GROUP_SIZE; i++)
//   {
//     sumOfWorkSums += results[i];
//   }
//   printf("actual: %lf, GPU: %lf \n", precount, sumOfWorkSums);
// }

int countDataEntries()
{
  FILE *file = fopen(FILE_NAME, "r");
  float num;
  int count = 0;
  while(fscanf(file, "%f", &num) > 0)
  {
    count++;
  }
  return count;
}

void storeDataToProcess(float* data)
{
	FILE *file = fopen(FILE_NAME, "r");
	float num;
  int i = 0;
	while(fscanf(file, "%f" ,&num) > 0)
	{
		data[i] = num;
    i++;
	}
	fclose(file);
}

void testPrintData(float* data)
{
  int i;
  for(i=1; i < data_size + 1; i++)
  {
    printf("%d : %f \n", i, data[i-1]);
  }
}

float seq_average(float* data)
{
  int i;
  float accumulate = 0;
  //START MEASUREMENT HERE
  for(i = 0; i < data_size + 1; i++)
  {
    accumulate += data[i];
  }
  printf("sum : %f \n", accumulate);
  //STOP MEASUREMENT HERE
  return accumulate/data_size;
}

int main (int argc, char** argv)
{
  data_size = countDataEntries();

  printf("Count: %d \n",data_size);

  float data[data_size];
  float results[data_size];

	storeDataToProcess(data);
  //testPrintData(data);

  float average = 0.0;
  average = seq_average(data);

  printf("sequential avg = %lf \n", average);
  //DONE SEQUENTIAL--

  //START OPENCL CALCULATIONs
  int error;

  size_t global = data_size;

  //size_t WG_SIZE = WORK_GROUP_SIZE;
  size_t WG_SIZE = WORK_GROUP_SIZE;
  size_t valueSize;


  cl_device_id devices;
  error = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &devices, NULL);

  if(error != CL_SUCCESS)
  {
    printf("failed to create a device group");
    return EXIT_FAILURE;
  }

  //create a compute context with a gpu
  cl_context context = clCreateContext(0, 1, &devices, NULL, NULL, &error);
  if(!context)
  {
    printf("failed to create a context");
    return EXIT_FAILURE;
  }

  cl_command_queue commands = clCreateCommandQueue(context, devices, 0, &error);
  if(!commands)
  {
    printf("failed to create a command queue \n");
    return EXIT_FAILURE;
  }

  //create  program object for a context, load kernel code
  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&parallelSum_kernel, NULL, &error);
  if(!program)
  {
    printf("cant create opencl program");
    exit(1);
  }

  //error = clBuildProgram(program, 1, devices, NULL, NULL, NULL);
  error = clBuildProgram(program, 1, &devices, NULL, NULL, NULL);
  if(error != CL_SUCCESS)
  {
    size_t len;
    char buffer[2048];
    printf("failed to build program");
    exit(1);
  }

  cl_kernel kernel = clCreateKernel(program, "parallelSum", &error);
  if(!kernel || error != CL_SUCCESS)
  {
    printf("failed to create kernel \n");
    exit(1);
  }

  cl_mem input = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * data_size, NULL, NULL);
  cl_mem output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * data_size, NULL, NULL);

  if(!input || !output)
  {
    printf("cant create buffer");
    exit(1);
  }

  error = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(float) * data_size, data, 0, NULL, NULL);

  if(error != CL_SUCCESS)
  {
    printf("error failed to enqueu buffer to device");
    exit(1);
  }

  //set kernel arguments
  error = 0;
  error = clSetKernelArg(kernel, 0, sizeof(unsigned int) * data_size, NULL);
  error |= clSetKernelArg(kernel,1, sizeof(cl_mem), (void *)&input);
  error |= clSetKernelArg(kernel,2, sizeof(cl_mem), (void *)&output);

  if(error != CL_SUCCESS)
  {
    printf("failed to set arguments \n");
    exit(1);
  }

  //enqueue command to execute on device
  error = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &WG_SIZE, 0, NULL, NULL);
  if(error)
  {
    printf("failed to exe kernel %d \n", error);
    exit(1);
  }

  clFinish(commands);

  clReleaseMemObject(input);
  clReleaseMemObject(output);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);

  return 0;
}

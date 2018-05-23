#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define DATA_SIZE 1024
#define WORK_GROUP_SIZE 32

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

float data[DATA_SIZE];
float results[DATA_SIZE];

float precount = 0;

void GenerateData()
{
  int i;
  for(i = 0;i < DATA_SIZE; i++)
  {
    data[i] = 2.12312;
    precount += data[i];
  }
}

void readGPUResult(){
  int i;
  float sumOfWorkSums = 0.0;
  for(i = 0; i < DATA_SIZE/WORK_GROUP_SIZE; i++)
  {
    sumOfWorkSums += results[i];
  }
  printf("actual: %f, GPU: %f \n", precount, sumOfWorkSums);
}


int main (int argc, char** argv)
{
  unsigned int count = DATA_SIZE;
  int err;
  size_t global;
  size_t local;

  //get data
  GenerateData();
  printf("precount: %f \n", precount);

  //connect to a compute deivce
  cl_platform_id platform_id = NULL;
  cl_device_id device_id = NULL;   
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
  err = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_ALL, 1, 
            &device_id, &ret_num_devices);

  if(err != CL_SUCCESS)
  {
    printf("ERROR: failed to create a device group \n");
    return EXIT_FAILURE;
  }
  //////////////////////////////

  printf("pass 2 \n");

  //create a compute cl_context
  cl_context context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
  if(!context)
  {
    printf("Error: failed to reate a compute context \n");
    return EXIT_FAILURE;
  }


  printf("pass 3 \n");

  //create a command commands
  cl_command_queue commands = clCreateCommandQueue(context, device_id, 0, &err);
  if(!commands)
  {
    printf("error: failed to create a command commands! \n");
    return EXIT_FAILURE;
  }

  printf("pass 4 \n");


  //create compute cl_program
  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&parallelSum_kernel, NULL, &err);
  if(!program)
  {
    printf("error: failed to create a compute program! \n");
    return EXIT_FAILURE;
  }

  printf("pass 5 \n");

  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if(err != CL_SUCCESS)
  {
    //only if failed, do this
    size_t len;
    char buffer[2048];

    printf("error: failed to build program executable \n");
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);

    printf("%s\n", buffer);
    exit(1);
  }

  printf("pass X \n");
  cl_kernel kernel = clCreateKernel(program, "parallelSum", &err);
  if(!kernel || err != CL_SUCCESS)
  {
    printf("error: failed to create compute kernel! \n");
    exit(1);
  }

  printf("pass 6 \n");

  cl_mem input = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * count, NULL, NULL);
  cl_mem output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, NULL);

  if(!input || !output)
  {
    printf("failed to allocate device memory \n");
    exit(1);
  }

  printf("pass 6 \n");

  //write out data set to the clCreateBuffer
  err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(float) * count, data, 0, NULL, NULL);

  if(err != CL_SUCCESS)
  {
    printf("error failed to write to source array \n");
    exit(1);
  }

  printf("pass 7 \n");


  //set the arguments to our compute kernel
  err = 0;
  err = clSetKernelArg(kernel, 0, count*sizeof(unsigned int), NULL);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &input);
  err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &output);


  if(err != CL_SUCCESS)
  {
    printf("error: failed to set kernel arguments! %d \n", err);
    exit(1);
  }

  printf("pass 8 \n");

  //get the max work group size for executing the kernel on the device_id
  err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
  if(err != CL_SUCCESS)
  {
    printf("error: failed to retrieve kernel work group info %d \n",err);
    exit(1);
  }

  printf("pass 9 \n");

  // Execute the kernel over the entire range of our 1d input data set
  // using the maximum number of work group items for this device
  global = count;
  err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
  if(err)
  {
    printf("error: failed to execute kernel~ %d \n", err);
    return EXIT_FAILURE;
  }

  printf("pass 10 \n");

  //wait for the command commands to get serviced before reading back results
  clFinish(commands);

  //READ back the results
  err = clEnqueueReadBuffer(commands, output, CL_TRUE, 0, sizeof(float)*count, results, 0, NULL, NULL);
  if(err != CL_SUCCESS)
  {
    printf("error: failed to read output array %d \n", err);
    exit(1);
  }

  //print out Results obtained from the GPU
  readGPUResult();

  clReleaseMemObject(input);
  clReleaseMemObject(output);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);

  return 0;
}

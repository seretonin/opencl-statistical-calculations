//hi3
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
//include windowsopencl
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define WORK_GROUP_SIZE 64
#define FILE_NAME "dataset_50K.txt" //<-- I'll give you seg fault if I dont exist !
#define GPU "GeForce"

const char *parallelSum_kernel = "\n" \
"__kernel void parallelSum(__global const double* input, __global double* groupSum, __local double* localSum) \n" \
"{                                                                                  \n" \
"   uint localID   = get_local_id(0);                                                \n" \
"   uint globalID  = get_global_id(0);                                               \n" \
"   uint groupID = get_group_id(0);                                                 \n" \
"   uint groupSize = get_local_size(0);                                              \n" \
"                                                                                   \n" \
"   localSum[localID] = input[globalID];                                            \n" \
"                                                                                   \n" \
"   for(uint stride = groupSize / 2; stride > 0; stride /= 2)                      \n" \
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

double getTime()
{
  struct timeval t;
  double sec, msec;

  while(gettimeofday(&t, NULL) != 0);
  sec = t.tv_sec;
  msec = t.tv_usec;
  sec = sec + msec/1000000.0;

  return sec;
}



int countDataEntries()
{
  FILE *file = fopen(FILE_NAME, "r");
  double num;
  int count = 0;
  while(fscanf(file, "%lf", &num) > 0)
  {
    count++;
  }
  return count;
}

void storeDataToProcess(double* data)
{
	FILE *file = fopen(FILE_NAME, "r");
	double num;
  int i = 0;
	while(fscanf(file, "%lf" ,&num) > 0)
	{
		data[i] = num;
    //data[i] = i;
    i++;
	}
	fclose(file);
}

void testPrintData(double* data, int data_size)
{
  int i;
  for(i=1; i < data_size + 1; i++)
  {
    printf("%d : %lf \n", i, data[i-1]);
  }
}

double seq_average(double* data, int data_size)
{
  double t1 = 0.0;
  double t2 = 0.0;

  //START MEASUREMENT FOR SEQUENTIAL SUM / MEAN
  t1 = getTime();

        int i;
        double average = 0;
        double accumulate = 0;

        for(i = 0; i < data_size + 1; i++)
        {
          accumulate += data[i];
        }

        average = accumulate/data_size;

  t2 = getTime();
  //STOP MEASUREMENT FOR SEQ sum and mean
  printf("seq sum : %lf \n", accumulate);
  printf("time taken seq: %6.5f secs \n", (t2-t1));

  return average;
}

int main (int argc, char** argv)
{
  //count how many entries are there
  double* data;
  double* results;

  int data_size; 
  data_size = countDataEntries();

  printf("Count: %d \n",data_size);


  size_t globalSize = data_size;
  size_t localSize = WORK_GROUP_SIZE;

  if((data_size % localSize) != 0)
  {
    printf("data size is not divisible by workgroup size. Datasize is : %d, workgroup size: %d \n", data_size, WORK_GROUP_SIZE);
    exit(1);
  }

  //double data[data_size];
  data = malloc(data_size *sizeof(double));
  if(data == NULL)
  {
    printf("failed to malloc results \n");
  }

	storeDataToProcess(data);
  //testPrintData(data, data_size);

  double average = 0.0;
  average = seq_average(data,data_size);

  printf("sequential avg = %lf \n", average);
  //DONE SEQUENTIAL--

 //START OPENCL CALCULATIONs
  double t1 = 0.0;
  double t2 = 0.0;

  //For Reading the results from GPU
  double resultsFromGPU = 0;
  double averageFromGPU = 0;

  int error;

  int numberOfWorkGroup = data_size/localSize;

  printf("datasize: %d, workgroup: %d, numberofworkgroup: %d \n", data_size, WORK_GROUP_SIZE, numberOfWorkGroup);
  
  //holder for results from each workgroup
  //double results[numberOfWorkGroup];
  results = malloc(numberOfWorkGroup*sizeof(double));
  if(results == NULL)
  {
    printf("failed to malloc results \n");
  }

  //initilise result array to 0
  int i = 0;
  for(i = 0; i < numberOfWorkGroup; i++)
  {
    results[i] = 0.0;
  }

  cl_device_id devices;
  error = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &devices, NULL);

  if(error != CL_SUCCESS)
  {
    printf("failed to create a device group");
    return EXIT_FAILURE;
  }

  //create a compute context with a gpu
  cl_context context = clCreateContext(NULL, 1, &devices, NULL, NULL, &error);
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
 
    printf("Error: Failed to build program executable!\n");
    clGetProgramBuildInfo(program, devices, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
    printf("%s\n", buffer);
    exit(1);
  }


  cl_kernel kernel = clCreateKernel(program, "parallelSum", &error);
  if(!kernel || error != CL_SUCCESS)
  {
    printf("failed to create kernel \n");
    exit(1);
  }


  cl_mem input = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size*sizeof(double), NULL, NULL);
  cl_mem output = clCreateBuffer(context, CL_MEM_READ_ONLY, numberOfWorkGroup*sizeof(double), NULL, NULL);

  if(!input || !output)
  {
    printf("cant create buffer");
    exit(1);
  }

  error = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(double) * data_size, data, 0, NULL, NULL);
  error = clEnqueueWriteBuffer(commands, output, CL_TRUE, 0, sizeof(double) * numberOfWorkGroup, results, 0, NULL, NULL);

  if(error != CL_SUCCESS)
  {
    printf("error failed to enqueu buffer to device");
    exit(1);
  }


  //set kernel arguments
  error = 0;
  error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
  error |= clSetKernelArg(kernel,1, sizeof(cl_mem), &output);
  error |= clSetKernelArg(kernel,2, localSize*sizeof(double), NULL);

  if(error != CL_SUCCESS)
  {
    printf("failed to set arguments \n");
    exit(1);
  }

  //START PARALLEL MEASUREMENT-- includes, execution of kernel, reading of buffer, and summing of workgroup sums on host
  t1 = getTime(); 
  //enqueue command to execute on device
  error = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
  if(error != CL_SUCCESS)
  {
    printf("failed to exe kernel %d \n", error);
    exit(1);
  }

  clFinish(commands);

  error = clEnqueueReadBuffer(commands, output, CL_TRUE, 0, sizeof(double)*numberOfWorkGroup, results, 0, NULL, NULL);
  if(error)
  {
    printf("failed to read results \n");
    exit(1);
  }

  for(i = 0; i < numberOfWorkGroup; i++)
  {
    resultsFromGPU += results[i];
  }

  averageFromGPU = resultsFromGPU / data_size;

  t2 = getTime();

  printf("SUM :Results from GPU is %lf \n", resultsFromGPU);

  printf("AVG :Results from GPU is %lf \n", averageFromGPU);

  printf("GPU time taken: %6.5f secs \n", (t2-t1));


  clReleaseMemObject(input);
  clReleaseMemObject(output);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);

  return 0;
}

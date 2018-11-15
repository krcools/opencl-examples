// Modified from the examples created by Tim Mattson
// as found at https://github.com/HandsOnOpenCL/
// This work is licensed under the Creative Commons Attribution 3.0 Unported License.
//
// To view a copy of this license, visit http ://creativecommons.org/licenses/by/3.0/
// or send a letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View, California, 94041, USA.

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <iterator>

#include <CL/cl.h>


const char *src =
"__kernel void matmat(                                          \n"
"	__global float *outputC,                                    \n"
"	int widthA,                                                 \n"
"	int heightA,                                                \n"
"	int widthB,                                                 \n"
"	int heightB,                                                \n"
"	__global float *inputA,                                     \n"
"	__global float *inputB                                      \n"
") {                                                            \n"
"	int row = get_global_id(1);                                 \n"
"	int col = get_global_id(0);                                 \n"
"	float sum = 0.0f;                                           \n"
"	for (int i = 0; i < widthA; i++) {                          \n"
"		sum += inputA[row*widthA + i] * inputB[i*widthB + col]; \n"
"	}                                                           \n"
"	outputC[row*widthB + col] = sum;                            \n"
"}                                                              \n"
;


int main() {

	const int wA = 1024;
	const int hA = 1024;
	const size_t dsA = wA*hA * sizeof(float);
	float *A = (float *)malloc(dsA);

	const int wB = 1024;
	const int hB = wA;
	const size_t dsB = wB*hB * sizeof(float);
	float *B = (float *)malloc(dsB);

	const int wC = wB;
	const int hC = hA;
	const size_t dsC = wC*hC * sizeof(float);
	float *C = (float *)malloc(dsC);
	
	cl_int err;

	cl_platform_id platform; err = clGetPlatformIDs(1, &platform, NULL); assert(err == 0);
	cl_device_id device; err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL); assert(err == 0);
	cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
	cl_context ctx = clCreateContext(cps, 1, &device, NULL, NULL, &err); assert(err == 0);
	cl_command_queue myqueue = clCreateCommandQueue(ctx, device, 0, &err); assert(err == 0);

	size_t max_work_group_size = -1;
	err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
	std::cout << "Maximum work group size for this device: " << max_work_group_size << std::endl;

	cl_uint max_work_item_dims = -1;
	err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &max_work_item_dims, NULL);
	std::cout << "Maximum work item dimensions: " << max_work_item_dims << std::endl;

	size_t max_work_item_sizes[20];
	err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_item_sizes), max_work_item_sizes, NULL);
	for (int i = 0; i < max_work_item_dims; ++i)
		std::cout << "Workgroup sz along dim " << i << ": " << max_work_item_sizes[i] << std::endl;



	cl_mem bufferA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, dsA, NULL, &err); assert(err == 0);
	cl_mem bufferB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, dsB, NULL, &err); assert(err == 0);
	cl_mem bufferC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, dsC, NULL, &err); assert(err == 0);

	err = clEnqueueWriteBuffer(myqueue, bufferA, CL_TRUE, 0, dsA, (void*)A, 0, NULL, NULL);	assert(err == 0);
	err = clEnqueueWriteBuffer(myqueue, bufferB, CL_TRUE, 0, dsB, (void*)B, 0, NULL, NULL);	assert(err == 0);

	cl_program myprog = clCreateProgramWithSource(ctx, 1, &src, NULL, &err); assert(err == 0);
	err = clBuildProgram(myprog, 0, NULL, NULL, NULL, NULL); assert(err == 0);
	cl_kernel mykernel = clCreateKernel(myprog, "matmat", &err); assert(err == 0);

	clSetKernelArg(mykernel, 0, sizeof(cl_mem), (void *)&bufferC);
	clSetKernelArg(mykernel, 1, sizeof(cl_int), (void *)&wA);
	clSetKernelArg(mykernel, 2, sizeof(cl_int), (void *)&hA);
	clSetKernelArg(mykernel, 3, sizeof(cl_int), (void *)&wB);
	clSetKernelArg(mykernel, 4, sizeof(cl_int), (void *)&hB);
	clSetKernelArg(mykernel, 5, sizeof(cl_mem), (void *)&bufferA);
	clSetKernelArg(mykernel, 6, sizeof(cl_mem), (void *)&bufferB);

	// You might want to change this to
	// get this example to run on your device
	size_t localws[2]  = { 64,  64 };
	size_t globalws[2] = { wC, hC };

	err = clEnqueueNDRangeKernel(myqueue, mykernel, 2, NULL, globalws, localws, 0, NULL, NULL);
	std::cout << err << std::endl;
	assert(err == 0);

	err = clEnqueueReadBuffer(myqueue, bufferC, CL_TRUE, 0, dsC, (void*)C, 0, NULL, NULL); assert(err == 0);

	float test = 0.0f;
	for (int i = 0; i < wA; ++i)
		test += A[i*hA + 0] * B[0 * wB + i];

	std::cout << test << std::endl;
	std::cout << C[0 * wC + 0] << std::endl;

	clReleaseProgram(myprog);
	clReleaseKernel(mykernel);
	clReleaseCommandQueue(myqueue);
	clReleaseContext(ctx);

	free(C);
	free(B);
	free(A);

	return 0;
}
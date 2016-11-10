// Modified from the examples created by Tim Mattson
// as found at https://github.com/HandsOnOpenCL/
// This work is licensed under the Creative Commons Attribution 3.0 Unported License.
//
// To view a copy of this license, visit http ://creativecommons.org/licenses/by/3.0/
// or send a letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View, California, 94041, USA.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/cl.h>

#define show(x) std::cout << #x << ": " << x << std::endl;

#define check(x)            \
{                           \
	cl_int _ciErrNum;       \
	_ciErrNum = x;          \
	assert(_ciErrNum == 0); \
}

const char * src =
"__kernel void vecadd(         \n"
"	__global float *a,         \n"
"	__global float *b,         \n"
"	__global float *c,         \n"
"	__const unsigned int n)    \n"
"{                             \n"
"	int id = get_global_id(0); \n"
"	if(id < n)                 \n"
"		c[id] = a[id] + b[id]; \n"
"}                             \n"
;

int main() {

	cl_int err;

	cl_platform_id platform;
	cl_device_id device;
	cl_context ctx;
	cl_command_queue myqueue;

	// Boilerplate...
	err = clGetPlatformIDs(1, &platform, NULL);
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
	cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
	ctx = clCreateContext(cps, 1, &device, NULL, NULL, &err);
	myqueue = clCreateCommandQueue(ctx, device, 0, &err);

	// Create data on the host
	const unsigned int n = 1000;
	float *h_a = (float *)malloc(sizeof(float) * n);
	float *h_b = (float *)malloc(sizeof(float) * n);
	float *h_c = (float *)malloc(sizeof(float) * n);
	for (int i = 0; i < n; i++) h_a[i] = h_b[i] = 0.5f*i;

	// Allocate memory on the device
	cl_mem d_a = clCreateBuffer(ctx, CL_MEM_READ_ONLY, n * sizeof(float), NULL, &err);
	cl_mem d_b = clCreateBuffer(ctx, CL_MEM_READ_ONLY, n * sizeof(float), NULL, &err);
	cl_mem d_c = clCreateBuffer(ctx, CL_MEM_READ_ONLY, n * sizeof(float), NULL, &err);

	// Initialise on device memory
	err = clEnqueueWriteBuffer(myqueue, d_a, CL_TRUE, 0, n * sizeof(float), (void*)h_a, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(myqueue, d_b, CL_TRUE, 0, n * sizeof(float), (void*)h_b, 0, NULL, NULL);
	// c does not need to be initialised since it is where results are written to

	cl_program myprog = clCreateProgramWithSource(ctx, 1, &src, NULL, &err);
	err = clBuildProgram(myprog, 0, NULL, NULL, NULL, NULL);

	cl_kernel mykernel = clCreateKernel(myprog, "vecadd", &err);

	clSetKernelArg(mykernel, 0, sizeof(cl_mem), (void *)&d_a);
	clSetKernelArg(mykernel, 1, sizeof(cl_mem), (void *)&d_b);
	clSetKernelArg(mykernel, 2, sizeof(cl_mem), (void *)&d_c);
	clSetKernelArg(mykernel, 3, sizeof(unsigned int), (void *)&n);

	size_t localSize = 64;
	size_t globalSize = (size_t)ceil(n / (float)localSize) * localSize;

	err = clEnqueueNDRangeKernel(myqueue, mykernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
	err = clEnqueueReadBuffer(myqueue, d_c, CL_TRUE, 0, n*sizeof(float), (void*)h_c, 0, NULL, NULL);

	float maxerr = 0.0f;
	for (int i = 0; i < n; i++) {
		float newerr = fabs(h_a[i] + h_b[i] - h_c[i]);
		if (newerr > maxerr)
			maxerr = newerr;
	}
	printf("Maximum error: %f\n", maxerr);

	clReleaseKernel(mykernel);
	clReleaseProgram(myprog);
	clReleaseMemObject(d_c);
	clReleaseMemObject(d_b);
	clReleaseMemObject(d_a);
	clReleaseCommandQueue(myqueue);
	clReleaseContext(ctx);
	clReleaseDevice(device);

	free(h_c);
	free(h_b);
	free(h_a);

	return 0;
}
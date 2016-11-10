#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

void exitmsg(const char *msg)
{ printf("%s\n", msg); exit(EXIT_FAILURE); }
void check(cl_int status, const char *msg)
{ if (status != CL_SUCCESS) exitmsg(msg); }

int output_device_info(cl_device_id device_id)
{
    int err;
    cl_device_type device_type;
    cl_uint comp_units;
    cl_char vendor_name[1024] = {0};
    cl_char device_name[1024] = {0};
    cl_uint max_work_itm_dims;
    size_t max_wrkgrp_size;
    size_t *max_loc_size;

    err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), &device_name, NULL);
	check(err, "Error: Failed to access device name!");
    printf(" \n Device is  %s ",device_name);

    err = clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);
	check(err, "Error: Failed to access device type information!");
    if(device_type  == CL_DEVICE_TYPE_GPU)
       printf(" GPU from ");
    else if (device_type == CL_DEVICE_TYPE_CPU)
       printf("\n CPU from ");
    else 
       printf("\n non  CPU or GPU processor from ");

    err = clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(vendor_name), &vendor_name, NULL);
	check(err, "Error: Failed to access device vendor name!");
    printf(" %s ",vendor_name);

    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &comp_units, NULL);
	check(err, "Error: Failed to access device number of compute units !");
    printf(" with a max of %d compute units \n",comp_units);

    err = clGetDeviceInfo( device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &max_work_itm_dims, NULL);
	check(err, "Error: Failed to get device Info (CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS)!");
    max_loc_size = (size_t*)malloc(max_work_itm_dims * sizeof(size_t));
    if(max_loc_size == NULL){ printf(" malloc failed\n"); return EXIT_FAILURE; }
    err = clGetDeviceInfo( device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, max_work_itm_dims* sizeof(size_t), max_loc_size, NULL);
	check(err, "Error: Failed to get device Info (CL_DEVICE_MAX_WORK_ITEM_SIZES)!");
    err = clGetDeviceInfo( device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_wrkgrp_size, NULL);
	check(err, "Error: Failed to get device Info (CL_DEVICE_MAX_WORK_GROUP_SIZE)");

   printf("work group, work item information");
   printf("\n max loc dim ");
   for(size_t i=0; i< max_work_itm_dims; i++)
     printf(" %d ",(int)(*(max_loc_size+i)));
   printf("\n");
   printf(" Max work group size = %d\n",(int)max_wrkgrp_size);

    return CL_SUCCESS;

}

int main(int argc, char *argv[])
{
	cl_int err;

	cl_platform_id platform;
	cl_device_id device;
	cl_context ctx;
	cl_command_queue myqueue;

	err = clGetPlatformIDs(1, &platform, NULL);
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);

	output_device_info(device);

	return 0;

}

#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h> 
#include <omp.h>
#include "FreeImage.h"
#include <sys/stat.h>
#include <time.h>
#include <math.h>

#include <unistd.h>
#include <ctype.h>

#define MAX_SOURCE_SIZE	16384

void printPlatformsInfo(cl_device_id *devices, cl_uint num_devices);
void checkStatus(cl_int status, char *location);

struct Color {
   unsigned char R;
   unsigned char G;
   unsigned char B;
};  


int main(int argc, char *argv[]) {    

    /*************************************/
    /*      PARSE ARGUMENTS              */    
    /*************************************/

    int K = 64;
    int I = 50;

    char *inputFile = NULL;
    char *outputFile = "compressed.png";

    char flag;
    while ((flag = getopt (argc, argv, "K:I:")) != -1) {
        switch (flag) {
            case 'K':
                K = atoi(optarg);
                if (K <= 1) {
                    fprintf(stderr, "Option -%c requires a positive number as argument.\n", optopt);
                }
                break;
                exit(1);
            case 'I':
                I = atoi(optarg);
                if (I <= 1) {
                    fprintf(stderr, "Option -%c requires a positive number as argument.\n", optopt);
                }
                break;
                exit(1);
            case '?':
                if (optopt == 'K' || optopt == 'I') {
                    fprintf(stderr, "Option -%c requires an argument.\n", optopt);
                } 
                else {
                    fprintf(stderr, "Unknown option `-%c'.\n", optopt);
                }
                exit(1);
            default:
                exit(1);
        }
    }
    
    int nonOptArgs = argc - optind;
    if (nonOptArgs == 1) {
        inputFile = argv[optind];
    } 
    else if (nonOptArgs == 2) {
        inputFile = argv[optind];
        outputFile = argv[optind+1];
    }
    else {
        fprintf(stderr, "Usage: ./gpu input_file output_file [-K clusters] [-I iterations]\n");
        exit(1);
    }


    srand(time(NULL));   
    cl_int status;

    /*************************************/
    /*      LOAD IMAGE                    */    
    /*************************************/

	FIBITMAP *imageBitmap = FreeImage_Load(FIF_PNG, inputFile, 0);
    // Convert to 32-bit image
    FIBITMAP *imageBitmap32 = FreeImage_ConvertTo32Bits(imageBitmap);
	
    // Get image dimensions
    int width = FreeImage_GetWidth(imageBitmap32);
	int height = FreeImage_GetHeight(imageBitmap32);
	int pitch = FreeImage_GetPitch(imageBitmap32);

    /*************************************/
    /*      ALLOCATE MEMORY ON HOST      */    
    /*************************************/
	
    // Prepare room for a raw data copy of the image
    unsigned char *imageIn= (unsigned char *)malloc(height * pitch * sizeof(unsigned char));
    // Extract raw data from the image
	FreeImage_ConvertToRawBits(imageIn, imageBitmap32, pitch, 32, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, TRUE);
    // Free source image data
	FreeImage_Unload(imageBitmap32);
	FreeImage_Unload(imageBitmap);


    int *c = malloc(width * height * sizeof(int));              // cluster number for each pixel
    struct Color *centroids = malloc(K * sizeof(struct Color)); // centroids (B, G, R)
    int *clusterCount = calloc(K * 4, sizeof(int));            // (Rsum, Gsum, Bsum, pixelCount) for each cluster
    int *randIndexes = malloc(K * sizeof(int));                


    // Initialize centroids - Randomly assign pixels 
    for(int i = 0; i < K; i++) {
        int y = rand() % (height - 2);
        int x = rand() % (width - 2);
        centroids[i].R = imageIn[(y*width+x)*4+2];
        centroids[i].G = imageIn[(y*width+x)*4+1];
        centroids[i].B = imageIn[(y*width+x)*4];
    }


    /*************************************/
    /*      DELITEV DELA                 */    
    /*************************************/

    // Kernel 1 
    size_t localItemSize = 256;
    size_t numGroups = (((height * width) - 1) / localItemSize + 1);
    size_t globalItemSize = numGroups * localItemSize;

    // Kernel 2
    size_t globalItemSize2 = K; 
    size_t localItemSize2 = K; 


    /*************************************/
    /*      READ KERNEL SOURCE           */    
    /*************************************/

    FILE *fp;
    char *sourceStr;
    size_t sourceSize;

    fp = fopen("kernels.cl", "r");
    if (!fp) {
		fprintf(stderr, "Error opening kernel.cl\n");
        exit(1);
    }
    sourceStr = (char*)malloc(MAX_SOURCE_SIZE);
    sourceSize = fread(sourceStr, 1, MAX_SOURCE_SIZE, fp);
	sourceStr[sourceSize] = '\0';
    fclose(fp);

    
    /*************************************/
    /*   DISCOVER AVAILABLE PLATFORMS    */    
    /*************************************/
 
    cl_platform_id	platforms[10];
    cl_uint	numOfPlatforms;
	char *buf;
	size_t bufLen;
	status = clGetPlatformIDs(10, platforms, &numOfPlatforms);
    checkStatus(status, "clGetPlatformIDs");

	
	cl_device_id devices[10];
	cl_uint	numOfDevices;
	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 10,	
						 devices, &numOfDevices);	
    checkStatus(status, "clGetDeviceIDs");


    // printPlatformsInfo(devices, numOfDevices);
    
    // select device
    const int deviceID = 0;

    /*************************************/
    /*   CREATE A CONTEXT                */    
    /*************************************/

    cl_context context = clCreateContext(NULL, 1, &devices[deviceID], NULL, NULL, &status);
    checkStatus(status, "clCreateContext");


    /*************************************/
    /*   CREATE A COMMAND QUEUE          */    
    /*************************************/

    cl_command_queue commandQueue = clCreateCommandQueue(context, devices[deviceID], 0, &status);
    checkStatus(status, "clCreateCommandQueue");



    /*************************************/
    /*   CREATE DEVICE BUFFERS           */    
    /*************************************/

    cl_mem imageIn_d = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
									  height * pitch * sizeof(unsigned char), imageIn, &status);
    checkStatus(status, "clCreateBuffer");


    cl_mem c_d = clCreateBuffer(context, CL_MEM_READ_WRITE, width * height * sizeof(int), NULL, &status);
    checkStatus(status, "clCreateBuffer");

    cl_mem centroids_d = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, K * sizeof(struct Color), centroids, &status);
    checkStatus(status, "clCreateBuffer");
        
    cl_mem clusterCount_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 4 * K * sizeof(int), NULL, &status);
    checkStatus(status, "clCreateBuffer");



    /*************************************/
    /*   CREATE PROGRAM OBJECT           */    
    /*************************************/

    cl_program program = clCreateProgramWithSource(context,	1, (const char **)&sourceStr, NULL, &status);												
    checkStatus(status, "clCreateProgramWithSource");

    /*************************************/
    /*   BUILD PROGRAM                   */    
    /*************************************/

    // Build program
    char buildArgs[64];
    sprintf(buildArgs, "-DK=%d", K);
    status = clBuildProgram(program, 1, &devices[deviceID], buildArgs, NULL, NULL);

    // Log kernel compilation errors
    if (status == CL_BUILD_PROGRAM_FAILURE) {
        size_t logSize;
        clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        char *log = (char *) malloc(logSize);
        clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
        printf("%s\n", log);
    }

  
    /*************************************/
    /*   COMPILE KERNELS                 */    
    /*************************************/

    cl_kernel kernel = clCreateKernel(program, "assignToCluster", &status);
    checkStatus(status, "clCreateKernel");

    cl_kernel kernel2 = clCreateKernel(program, "updateCentroids", &status);
    checkStatus(status, "clCreateKernel");


    /*************************************/
    /*   SET KERNEL ARGUMENTS            */    
    /*************************************/
 
    // kernel1
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&imageIn_d);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&c_d);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&centroids_d);
    status |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&clusterCount_d);
    status |= clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&height);
    status |= clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&width);
    checkStatus(status, "clSetKernelArg");

    // kernel2
    status = clSetKernelArg(kernel2, 0, sizeof(cl_mem), (void *)&centroids_d);
    status |= clSetKernelArg(kernel2, 2, sizeof(cl_mem), (void *)&c_d);
    status |= clSetKernelArg(kernel2, 4, sizeof(cl_mem), (void *)&imageIn_d);
    checkStatus(status, "clSetKernelArg");


    double startTime = omp_get_wtime();


    /*************************************/
    /*   RUN                             */    
    /*************************************/

    for (int i = 0; i < I; i++) {    

        // Reset clusterCount
        clusterCount_d = clCreateBuffer(context, CL_MEM_READ_WRITE, K * 4 * sizeof(int), NULL, &status);
        checkStatus(status, "clCreateBuffer");
        status = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&clusterCount_d);
        checkStatus(status, "clSetKernelArg");
        status = clSetKernelArg(kernel2, 1, sizeof(cl_mem), (void *)&clusterCount_d);
        checkStatus(status, "clSetKernelArg");

        status = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,						
                                    &globalItemSize, &localItemSize, 0, NULL, NULL);	
        checkStatus(status, "clEnqueueNDRangeKernel 1");

        
        // // Generate sequence of random pixel indexes (for fixing empty clusters)
        for (int j = 0; j < K; j++) {
            randIndexes[j] = rand() % (width * height - 2);
        }
        cl_mem randIndexes_d = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
									  K * sizeof(int), randIndexes, &status);
        checkStatus(status, "clCreateBuffer");
        status = clSetKernelArg(kernel2, 3, sizeof(cl_mem), (void *)&randIndexes_d);
        checkStatus(status, "clSetKernelArg");                    


        status = clEnqueueNDRangeKernel(commandQueue, kernel2, 1, NULL, &globalItemSize2, &localItemSize2, 0, NULL, NULL);	
        checkStatus(status, "clEnqueueNDRangeKernel kernel 2");
    }

    /*************************************/
    /*   READ RESULTS BACK TO HOST       */    
    /*************************************/
																	
    // Read result from device
    status = clEnqueueReadBuffer(commandQueue, c_d, CL_TRUE, 0, width * height * sizeof(int), c, 0, NULL, NULL);				
    checkStatus(status, "clEnqueueReadBuffer");

    status = clEnqueueReadBuffer(commandQueue, centroids_d, CL_TRUE, 0, K * sizeof(struct Color), centroids, 0, NULL, NULL);				
    checkStatus(status, "clEnqueueReadBuffer");

    status = clEnqueueReadBuffer(commandQueue, clusterCount_d, CL_TRUE, 0, K * 4 * sizeof(int), clusterCount, 0, NULL, NULL);				
    checkStatus(status, "clEnqueueReadBuffer");


    /*************************************/
    /*   CREATE OUTPUT IMAGE             */    
    /*************************************/

    unsigned char *imageOut = (unsigned char *) malloc(height * pitch * sizeof(unsigned char));
    for (int i = 0; i < width * height; i++) {
        int cluster = c[i];
        imageOut[i*4+3] = 255; 
        imageOut[i*4+2] = centroids[cluster].R; 
        imageOut[i*4+1] = centroids[cluster].G; 
        imageOut[i*4] = centroids[cluster].B; 
    }

    printf("Input file: %s\n", inputFile);
    printf("Output file: %s\n", outputFile);
    printf("I: %d K: %d\n", I, K);
    printf("Time: %.3fs\n", omp_get_wtime() - startTime);


    // Save image     
    FIBITMAP *dst = FreeImage_ConvertFromRawBits(imageOut, width, height, pitch,
		32, 0xFF, 0xFF, 0xFF, TRUE);
	FreeImage_Save(FIF_PNG, dst, outputFile, 0);


    /*************************************/
    /*  CALCULATE FILE SIZE REDUCTION   */    
    /*************************************/

    struct stat st;
    stat(inputFile, &st);
    int inSize =  (int) (st.st_size / 1024); 
    stat(outputFile, &st);
    int outSize =  (int) (st.st_size / 1024);     
    printf("File size reduction: %.2f%\n", 100 *  (1 - (double) outSize  / inSize));


    /*************************************/
    /*   CLEANUP                         */    
    /*************************************/

    clFlush(commandQueue);
    clFinish(commandQueue);
    if (kernel) clReleaseKernel(kernel);
    if (program) clReleaseProgram(program);
    if (imageIn_d) clReleaseMemObject(imageIn_d);
    if (c_d) clReleaseMemObject(c_d);
    if (centroids_d) clReleaseMemObject(centroids_d);

    if (commandQueue) clReleaseCommandQueue(commandQueue);
    if (context) clReleaseContext(context);
	
    free(imageIn);	
    free(c);
    free(centroids);
    free(randIndexes);

    return 0;
}


    

/*   helper functions for OpenCL    */    


char const* getErrorString(cl_int error) {
    switch(error){
        // run-time and JIT compiler errors
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

        // compile-time errors
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_commandQueue";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

        // extension errors
        case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        default: return "Unknown OpenCL error";
        }
}

void checkStatus(cl_int status, char *location) {
    if (status != CL_SUCCESS) {
        printf("Error @ %s ... %s\n", location, getErrorString(status));
        exit(1);
    }
}


void printPlatformsInfo(cl_device_id *devices, cl_uint num_devices) {

    char buffer[10000];
    cl_uint buf_uint;
    cl_ulong buf_ulong;
    size_t buf_sizet;

    printf("=== OpenCL devices: ===\n");
    for (int i=0; i<num_devices; i++)
    {
        printf("  -- The device with the index %d --\n", i);
        clGetDeviceInfo(devices[i],
                        CL_DEVICE_NAME,
                        sizeof(buffer),
                        buffer,
                        NULL);
        printf("  CL_DEVICE_NAME = %s\n", buffer);

        clGetDeviceInfo(devices[i],
                        CL_DEVICE_VENDOR,
                        sizeof(buffer),
                        buffer,
                        NULL);
        printf("  CL_DEVICE_VENDOR = %s\n", buffer);

        clGetDeviceInfo(devices[i],
                        CL_DEVICE_MAX_CLOCK_FREQUENCY,
                        sizeof(buf_uint),
                        &buf_uint,
                        NULL);
        printf("  CL_DEVICE_MAX_CLOCK_FREQUENCY = %u\n",
               (unsigned int)buf_uint);
    
        clGetDeviceInfo(devices[i],
                        CL_DEVICE_MAX_COMPUTE_UNITS,
                        sizeof(buf_uint),
                        &buf_uint,
                        NULL);
        printf("  CL_DEVICE_MAX_COMPUTE_UNITS = %u\n",
               (unsigned int)buf_uint);

        clGetDeviceInfo(devices[i],
                        CL_DEVICE_MAX_WORK_GROUP_SIZE,
                        sizeof(buf_sizet),
                        &buf_sizet,
                        NULL);
        printf("  CL_DEVICE_MAX_WORK_GROUP_SIZE = %u\n",
               (unsigned int)buf_sizet);
               
        clGetDeviceInfo(devices[i],
                        CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                        sizeof(buf_uint),
                        &buf_uint,
                        NULL);
        printf("  CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS = %u\n",
               (unsigned int)buf_uint);
        
        size_t workitem_size[3];
        clGetDeviceInfo(devices[i],
                        CL_DEVICE_MAX_WORK_ITEM_SIZES,
                        sizeof(workitem_size),
                        &workitem_size,
                        NULL);
        printf("  CL_DEVICE_MAX_WORK_ITEM_SIZES = %u, %u, %u \n",
               (unsigned int)workitem_size[0],
               (unsigned int)workitem_size[1],
               (unsigned int)workitem_size[2]);

        clGetDeviceInfo(devices[i],
                        CL_DEVICE_GLOBAL_MEM_SIZE,
                        sizeof(buf_ulong),
                        &buf_ulong,
                        NULL);
        printf("  CL_DEVICE_GLOBAL_MEM_SIZE = %u\n",
               (unsigned int)buf_ulong);
        
        clGetDeviceInfo(devices[i],
                        CL_DEVICE_LOCAL_MEM_SIZE,
                        sizeof(buf_ulong),
                        &buf_ulong,
                        NULL);
        printf("  CL_DEVICE_LOCAL_MEM_SIZE = %u\n",
               (unsigned int)buf_ulong);
               
    }
}





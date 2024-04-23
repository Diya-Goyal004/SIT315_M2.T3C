#include <mpi.h> // MPI header for distributed computing
#include <CL/cl.h> // OpenCL header 
#include <stdio.h> // Standard I/O functions
#include <stdlib.h> // Standard library functions (malloc, free, etc.)
#include <string.h> // String functions
#include <time.h> // Time functions for seeding random generator

#define MAX_SOURCE_SIZE 0x100000 // Maximum size for OpenCL kernel source code

// OpenCL kernel for quicksort
const char* quicksort_kernel_source =
"__kernel void quicksort(__global int* arr, int left, int right) {\n"
"    int i = left;\n"
"    int j = right;\n"
"    int pivot = arr[(left + right) / 2];\n" // Pivot for partitioning
"    while (i <= j) {\n"
"        while (arr[i] < pivot) i++;\n" // Move from left to right until finding an element larger than pivot
"        while (arr[j] > pivot) j--;\n" // Move from right to left until finding an element smaller than pivot
"        if (i <= j) {\n"
"            int tmp = arr[i];\n" // Swap the elements if i and j indices cross
"            arr[i] = arr[j];\n"
"            arr[j] = tmp;\n"
"            i++;\n"
"            j--;\n"
"        }\n"
"    }\n"
"    if (left < j) quicksort(arr, left, j); // Recursive quicksort for left partition
"    if (i < right) quicksort(arr, i, right); // Recursive quicksort for right partition
"}\n";

// Function to perform quicksort with MPI and OpenCL
void quicksort_mpi_opencl(int* arr, int n, int rank, int size, cl_context context, cl_command_queue queue, cl_program program) {
    // Divide the array into chunks for each MPI process
    int local_n = n / size;
    int* local_arr = (int*)malloc(local_n * sizeof(int));
    
    // Scatter the data among all MPI processes
    MPI_Scatter(arr, local_n, MPI_INT, local_arr, local_n, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Create the OpenCL kernel
    cl_int ret;
    cl_kernel kernel = clCreateKernel(program, "quicksort", &ret);
    if (ret != CL_SUCCESS) { // Check for errors in kernel creation
        printf("Error creating kernel\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Create a buffer in OpenCL memory
    cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, local_n * sizeof(int), NULL, &ret);
    if (ret != CL_SUCCESS) { // Check for errors in buffer creation
        printf("Error creating buffer\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Copy data from local array to OpenCL buffer
    ret = clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, local_n * sizeof(int), local_arr, 0, NULL, NULL);
    
    // Set kernel arguments
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
    ret |= clSetKernelArg(kernel, 1, sizeof(int), &((int){0}));
    ret |= clSetKernelArg(kernel, 2, sizeof(int), &((int){local_n - 1}));
    
    // Execute the OpenCL kernel
    size_t global_work_size[1] = {local_n}; // Work size for OpenCL
    ret = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
    clFinish(queue); // Wait for the kernel to complete
    
    // Read the sorted data back to the local array
    ret = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, local_n * sizeof(int), local_arr, 0, NULL, NULL);
    
    // Gather the sorted data from all MPI processes
    MPI_Gather(local_arr, local_n, MPI_INT, arr, local_n, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Clean up OpenCL resources
    clReleaseMemObject(buffer);
    clReleaseKernel(kernel);
    free(local_arr); // Free the memory allocated for local array
}

// Main function
int main(int argc, char** argv) {
    int rank, size; // MPI rank and size variables
    MPI_Init(&argc, &argv); // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the current MPI rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the total number of MPI processes
    
    int n = 10000; // Size of the array to be sorted
    int* arr = NULL; // Pointer for the array to be sorted
    
    double start_time, end_time; // Timing variables
    
    start_time = MPI_Wtime(); // Start timing
    
    if (rank == 0) { // Master process
        arr = (int*)malloc(n * sizeof(int)); // Allocate memory for the array
        srand((unsigned)time(NULL)); // Seed the random number generator
        for (int i = 0; i < n; i++) { // Initialize array with random numbers
            arr[i] = rand() % 10000; // Random values between 0 and 9999
        }
    }
    
    // OpenCL setup
    cl_int ret;
    cl_platform_id platform_id;
    ret = clGetPlatformIDs(1, &platform_id, NULL); // Get OpenCL platform ID
    if (ret != CL_SUCCESS) { // Check for errors in getting platform ID
        printf("Error getting platform\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    cl_device_id device_id;
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL); // Get OpenCL device ID
    if (ret != CL_SUCCESS) { // Check for errors in getting device ID
        printf("Error getting device\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Create OpenCL context and command queue
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    if (ret != CL_SUCCESS) { // Check for errors in creating context
        printf("Error creating context\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    cl_command_queue queue = clCreateCommandQueue(context, device_id, 0, &ret); // Create OpenCL command queue
    if (ret != CL_SUCCESS) { // Check for errors in creating queue
        printf("Error creating command queue\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Create OpenCL program with the provided kernel source
    cl_program program = clCreateProgramWithSource(context, 1, &quicksort_kernel_source, NULL, &ret);
    if (ret != CL_SUCCESS) { // Check for errors in creating program
        printf("Error creating program\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Build the OpenCL program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL); // Build the OpenCL program
    if (ret != CL_SUCCESS) { // Check for errors in building program
        printf("Error building program\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Perform quicksort using MPI and OpenCL
    quicksort_mpi_opencl(arr, n, rank, size, context, queue, program);
    
    end_time = MPI_Wtime(); // End timing
    
    if (rank == 0) { // If this is the master process
        printf("Execution time: %.6f seconds\n", end_time - start_time); // Display total execution time
        
        // Check if the array is sorted
        int sorted = 1;
        for (int i = 1; i < n; i++) {
            if (arr[i - 1] > arr[i]) { // If any element is out of order
                sorted = 0;
                break;
            }
        }
        
        if (sorted) {
            printf("Array is sorted\n"); // Indicate if the array is sorted
        } else {
            printf("Array is not sorted\n"); // Indicate if the array is not sorted
        }
        
        free(arr); // Free memory for the array
    }
    
    // Clean up OpenCL resources
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    MPI_Finalize(); // Finalize MPI
    
    return 0; // Exit the program
}

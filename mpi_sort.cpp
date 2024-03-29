#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <mpi.h>

using namespace std;

// Function to perform quicksort algorithm
void quickSort(vector<int>& arr, int left, int right) {
    int i = left, j = right;
    int pivot = arr[(left + right) / 2]; // Choose pivot element

    // Partitioning step
    while (i <= j) {
        // Find element on left that should be on right
        while (arr[i] < pivot)
            i++;
        // Find element on right that should be on left
        while (arr[j] > pivot)
            j--;

        // Swap elements and move indices
        if (i <= j) {
            swap(arr[i], arr[j]);
            i++;
            j--;
        }
    };

    // Recursively sort two subarrays
    if (left < j)
        quickSort(arr, left, j); // Sort left subarray
    if (i < right)
        quickSort(arr, i, right); // Sort right subarray
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); // Initialize MPI environment
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get rank of current process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get total number of processes

    if (argc != 2) { // Check if array size is provided as argument
        if (rank == 0)
            cout << "Usage: mpirun -n <num_procs> ./mpi_sort <array_size>" << endl;
        MPI_Finalize(); // Finalize MPI environment
        return 1;
    }

    int n = atoi(argv[1]); // Convert array size argument to integer
    vector<int> arr(n); // Declare vector to hold array elements

    // Initialize random array on root process
    if (rank == 0) {
        srand(time(NULL)); // Seed random number generator
        for (int i = 0; i < n; ++i)
            arr[i] = rand() % 1000; // Generate random numbers between 0 and 999
    }

    // Broadcast array to all processes
    MPI_Bcast(arr.data(), n, MPI_INT, 0, MPI_COMM_WORLD);

    // Perform parallel quicksort
    double start_time = MPI_Wtime(); // Start timer
    quickSort(arr, 0, n - 1); // Sort the array
    double end_time = MPI_Wtime(); // End timer

    // Print sorted array on root process
    if (rank == 0) {
        cout << "Sorted array:" << endl;
        for (int i = 0; i < n; ++i)
            cout << arr[i] << " "; // Print sorted elements
        cout << endl;

        cout << "Execution time: " << end_time - start_time << " seconds" << endl; // Print execution time
    }

    MPI_Finalize(); // Finalize MPI environment
    return 0;
}

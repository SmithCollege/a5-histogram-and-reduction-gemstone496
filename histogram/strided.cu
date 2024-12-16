#include <stdio.h>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <cmath>
#include <random>

#define SIZE 100000000
#define BLOCKS 4096
#define THREADS 1024
#define RUNS 100

/** function to calculate the scan on GPU */
__global__ void histogram(unsigned char *buffer,
			  long size,
			  unsigned int *histo){
  for ( long i = threadIdx.x + blockIdx.x*blockDim.x, stride = gridDim.x*blockDim.x;
	i < size;
	i += stride ) {
    unsigned char item = buffer[i];
    atomicAdd(&histo[item], 1);
  }

}

/** sets every character in vals (up to size `length`) to a random char (binomial distribution). does not allocate memory */
void generateVals(unsigned char* vals,
		  long length) {
  std::random_device rd;
  std::mt19937 gen(rd()); // the docs used this in the example but idk what it is. some kind of seed i'd guess?
  std::binomial_distribution<> dist(255, 0.5); // any char is valid game but i want some more common for realism

  for (int i = 0; i < length; i++) {
    vals[i] = dist(gen); // i could randomize this or i could choose not to
  }
}

/** what it says on the tag */
int main() {
  std::cout << "\nstrided" << SIZE; // record the size of the run for data collection

  // allocate input and output arrays
  int bins = pow(2, 8*sizeof(char)); // a bin for every possible char
  unsigned char *buffer, *in; unsigned int *histo, *reset, *out; // named vars are host, in/out are device

  buffer = (unsigned char*) malloc( SIZE * sizeof(char) );
  histo = (unsigned int*) malloc ( bins * sizeof(int) );
  reset = (unsigned int*) malloc ( bins * sizeof(int) );
  cudaMalloc(&in, SIZE * sizeof(char));
  cudaMalloc(&out, bins * sizeof(int));

  generateVals(buffer, SIZE);
  cudaMemcpy(in, buffer, SIZE * sizeof(char), cudaMemcpyHostToDevice);

  for (int i = 0; i < bins; i++) { // reset
    reset[i] = 0;
  }

  // run 100 times or whatever
  for (int i = 0; i < RUNS; i++) {

    cudaMemcpy(out, reset, bins * sizeof(int), cudaMemcpyHostToDevice);

    // time it
    const auto start{std::chrono::steady_clock::now()};
    histogram<<<BLOCKS, THREADS>>>(in, SIZE, out);
    cudaDeviceSynchronize();
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed{end - start};
    std::cout << "," << elapsed.count();

    // double check it
    // std::cerr << cudaGetErrorString(cudaGetLastError()) << "\n";

  }

  // check reasonable output
  cudaMemcpy(histo, out, bins * sizeof(int), cudaMemcpyDeviceToHost);
  int count = 0;
  for (int i = 0; i < bins; i++) {
    count += histo[i]; // how many chars did i find
  }
  if (count != SIZE) { std::cerr << "COUNT: " << count << "   EXP: " << SIZE << "\n"; }

  // free mem
  free(buffer);
  free(histo);
  free(reset);
  cudaFree(in);
  cudaFree(out);

  return 0;
}

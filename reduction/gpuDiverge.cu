#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <math.h>

#define SIZE 1000000
#define BLOCK_SIZE 1024
#define RUNS 5

// evaluate!! for function flexibility
__host__ __device__ int eval(int a, int b, char op){
  switch (op) {
    case 's': // sum
      return a+b;
    case 'p': // product
      return a*b;
    case 'l': // low
      return min(a, b);
    case 'h': // high
      return max(a, b);
  }

  return 0; // just throwing this in in case, who knows /shrug
}

__global__ void reduce(int *in, unsigned int size, char op, int *out){
  __shared__ int partialOp[BLOCK_SIZE];
  int t = threadIdx.x, dim = blockDim.x; // index for shared mem and length of blocks
  unsigned int gindex = t + blockIdx.x*dim; // index for in

  // load 2 values from global mem, and prereduce once. same # of polls from global mem, but saves shared mem by factor of 2.
  if (gindex*2 < size) { // never give me odd sizes T-T
    
    gindex*2 +1 != size ? partialOp[t] = eval(in[gindex*2], in[gindex*2 +1], op) : partialOp[t] = in[gindex*2]; // handle odd sizes

    for ( unsigned int stride = 1, remaining = (size+1) >> 1;
	  stride < dim;
	  stride <<= 2 ) {

      __syncthreads();
      if ( t % (stride*2) == 0 && t+stride < remaining) {
	partialOp[t] = eval(partialOp[t], partialOp[t +stride], op);
      }
      
    }
  
    if (t == 0) { out[blockIdx.x] = partialOp[0]; } // if i'm thread 0 set my output array's value
    
  }
  // this unironically makes less sense to me than the forwards/backwards version.
  // if you had just told me to do one implementation and not gone over why fwd/bkwd is more efficient i still would have done it...
}

int main() {
  std::cout << "\n" << "diverge" << SIZE; // record the size of the run for data collection

  int *in;
  in = (int*) malloc(SIZE*sizeof(int));
  // initialize inputs
  for (int i = 0; i < SIZE; i++) {
    in[i] = 1;
  }

  // run 100 times or whatever
  for (int i = 0; i < RUNS; i++) {
    // allocate mem and create vars. inside this loop because each run clobbers values
    unsigned int cachedSize = SIZE; // just initialize it so the calc goes correctly
    char op = 's';

    int *d_in, *out, *d_out;
    out = (int*) malloc( ceil(cachedSize/(2.*BLOCK_SIZE)) *sizeof(int) ); // meh    
    cudaMalloc(&d_in, SIZE*sizeof(int));
    cudaMemcpy(d_in, in, SIZE*sizeof(int), cudaMemcpyHostToDevice);

    const auto start{std::chrono::steady_clock::now()};

    do {
      unsigned int numBlocks = ceil( cachedSize / (2. *BLOCK_SIZE) ); // all this fancy work for just one op per cycle...
      cudaMalloc(&d_out, numBlocks * sizeof(int)); // allocate out (old out is new in, if second or later round)

      reduce<<<numBlocks, BLOCK_SIZE>>>(d_in, cachedSize, op, d_out); // calculate with sum
      cudaDeviceSynchronize(); // patience, little ones
      
      cudaFree(d_in); d_in = d_out; // free in and set it to the output
      cachedSize = numBlocks;
    }
    while (cachedSize > 1); // the do/while just made more sense to me intuitively than the for

    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed{end - start};
    std::cout << "," << elapsed.count();

    int exp = SIZE; // adjust based on appropriate function...
    cudaMemcpy(out, d_in, sizeof(int), cudaMemcpyDeviceToHost); // just the first one :)
    if (out[0] != exp) { std::cerr << "\nout " << out[0] << "    exp " << exp; }
    
    // free mem
    free(out);
    cudaFree(d_in); // no need to free d_in because it is freed at the end of the loop
  }

  free(in);

  return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <math.h>

#define SIZE 64
#define BLOCK_SIZE 64
#define RUNS 100

// evaluate!! for function flexibility
__device__ int eval(int a, int b, char op){
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

__global__ void reduce(int *in, int size, char op, int *out){
  __shared__ int partialOp[BLOCK_SIZE];
  int t = threadIdx.x; // index for temp
  int gindex = t + blockIdx.x*blockDim.x; // index for in

  // load 2 values from global mem, and prereduce once. same # of polls from global mem, but saves shared mem by factor of 2.
  if (gindex*2 < size) { // never give me odd sizes T-T
    
    gindex*2 +1 != size ? partialOp[t] = eval(in[gindex*2], in[gindex*2 +1], op) : partialOp[t] = in[gindex*2]; // handle odd sizes
    
    for ( unsigned int stride = 1;
	  stride <= blockDim.x;
	  stride *= 2 ) {

      __syncthreads();
      if ( t % (stride*2) == 0 && t+stride < size ) {
	partialOp[t] = eval(partialOp[t], partialOp[t +stride], op);
      }
      
    }
  
    if (t == 0) { out[blockIdx.x] = partialOp[0]; } // if i'm thread 0 set my output array's value
    
  }
  // this unironically makes less sense to me than the forwards/backwards version.
  // if you had just told me to do one implementation and not gone over why fwd/bkwd is more efficient i still would have done it...
}


int main() {
  std::cout << "\n" << SIZE; // record the size of the run for data collection
  
  // run 100 times or whatever
  for (int i = 0; i < RUNS; i++) {
    // allocate mem and create vars. inside this loop because each run clobbers values
    int *in, *d_in, *out, *d_out;
    
    in = (int*) malloc(SIZE*sizeof(int));
    out = (int*) malloc(SIZE*sizeof(int));
    cudaMalloc(&d_in, SIZE*sizeof(int));
    char op = 's';
    
    // initialize inputs
    for (int i = 0; i < SIZE; i++) {
      in[i] = 1;
    }
    cudaMemcpy(d_in, in, SIZE*sizeof(int), cudaMemcpyHostToDevice);

    int cachedSize = SIZE; // just initialize it so the calc goes correctly

    const auto start{std::chrono::steady_clock::now()};

    do {
      int numBlocks = ceil( (1.*cachedSize) / BLOCK_SIZE ); // all this fancy work just to cut one division operation per cycle...
      cudaMalloc(&d_out, numBlocks * sizeof(int)); // allocate out (old out is new in, if second or later round)

      reduce<<<numBlocks, BLOCK_SIZE>>>(d_in, cachedSize, op, d_out); // calculate with sum
      cudaDeviceSynchronize(); // patience, little ones
      
      cudaMemcpy(out, d_out, numBlocks*sizeof(int), cudaMemcpyDeviceToHost);
      
      cudaFree(d_in); d_in = d_out; // free in and set it to the output
      cachedSize = numBlocks;
      	
    }
    while (cachedSize > 1); // the do/while just made more sense to me intuitively than the for loop

    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed{end - start};
    std::cout << "," << elapsed.count();

    int exp = SIZE; // adjust based on appropriate function...
    cudaMemcpy(out, d_out, sizeof(int), cudaMemcpyDeviceToHost); // just the first one :)
    if (out[0] != exp) { std::cerr << "\nOUT: " << out[0] << "   EXP: " << exp; }
    
    // free mem
    free(in);
    free(out);
    cudaFree(d_in); // no need to free d_out because it is freed at the end of the loop
  }


  return 0;
}

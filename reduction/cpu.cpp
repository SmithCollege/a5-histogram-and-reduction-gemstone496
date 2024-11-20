#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <math.h>

#define SIZE 64
#define RUNS 1

// functions defined to pass into reduction
int sum(int a, int b) {
  return a+b;
}
int product(int a, int b) {
  return a*b;
}

// function to calculate the scan on GPU
int reduce(int *in, int size, int (*op)(int, int)){
  int out = in[0]; // begin. crash on empty inputs.

  for ( int i = 1; i < size; i++ ) {
    out = op(out, in[i]);
  }

  return out;
}

int main() {
  std::cout << "\n" << SIZE; // record the size of the run for data collection

  // allocate input and output arrays
  int* in, out;
  in = (int*) malloc(SIZE*sizeof(int));
  
  // run 100 times or whatever
  for (int i = 0; i < RUNS; i++) {
    // initialize inputs
    for (int j = 0; j < SIZE; j++) {
      in[j] = 1;
    }
    
    const auto start{std::chrono::steady_clock::now()};
    out = reduce(in, SIZE, &sum); // calculate with sum
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed{end - start};
    std::cout << "," << elapsed.count();

    if (out != SIZE) { std::cerr << "OUT: " << out << "   EXP: " << SIZE << "\n"; }
  }

  // free mem
  free(in);

  return 0;
}

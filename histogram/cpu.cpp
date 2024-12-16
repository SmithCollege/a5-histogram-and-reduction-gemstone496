#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <math.h>

#define SIZE 100000000
#define RUNS 100

// function to calculate the scan on GPU
void histogram(unsigned char *buffer,
			long size,
			unsigned int *histo){
  for ( int i = 0; i < size; i++ ) {
    unsigned char item = buffer[i];
    histo[item] += 1;
  }

}

void generateVals(unsigned char* vals,
		  long length) {
  for (int i = 0; i < length; i++) {
    vals[i] = 'a'; // i could randomize this or i could choose not to
  }

}

int main() {
  std::cout << "\ncpu" << SIZE; // record the size of the run for data collection

  // allocate input and output arrays
  unsigned char *buffer; unsigned int *histo;
  buffer = (unsigned char*) malloc( SIZE * sizeof(char) );
  histo = (unsigned int*) malloc ( pow(2, 8*sizeof(char)) * sizeof(int) ); // an int for every possible char

  generateVals(buffer, SIZE);
  
  // run 100 times or whatever
  for (int i = 0; i < RUNS; i++) {

    for (int j = 0; j < pow(2, 8*sizeof(char)); j++) { // reset
      histo[j] = 0;
    }

    // time it
    const auto start{std::chrono::steady_clock::now()};
    histogram(buffer, SIZE, histo); // calculate with sum
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed{end - start};
    std::cout << "," << elapsed.count();

    // verify that every letter was indeed 'a' as supplied
    if (histo['a'] != SIZE) { std::cerr << "OUT: " << histo['a'] << "   EXP: " << SIZE << "\n"; } // all of them are 'a's
  }

  // free mem
  free(buffer);
  free(histo);

  return 0;
}

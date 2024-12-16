#include <iostream>

#define RUNS 100


int main() {
  std::cout << "run";
  for (int i = 0; i < RUNS; i++) {
    std::cout << "," << i;
  }
}

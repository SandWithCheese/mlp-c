#include "inc/mlp.h"

// TODO
// 1. For each train data, we get n input
// 2.

int main() {
  NeuralNetwork nn;
  InitInputLayer(&nn, 3);
  InitHiddenLayer(&nn);
  AddHiddenLayer(&nn, 3, 0);
  AddHiddenLayer(&nn, 2, 1);
  AddHiddenLayer(&nn, 2, 1);
  PrintNeuralNetwork(nn);
  return 0;
}

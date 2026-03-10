#include "inc/mlp.h"
#include "inc/perceptron.h"

// TODO
// 1. For each train data, we get n input
// 2.

int main() {
  NeuralNetwork nn;
  InitInputLayer(&nn, 3);
  InitHiddenLayer(&nn);
  AddHiddenLayer(&nn, 3, RELU, UNIFORM);
  AddHiddenLayer(&nn, 2, RELU, UNIFORM);
  AddHiddenLayer(&nn, 2, RELU, UNIFORM);
  PrintNeuralNetwork(nn);
  return 0;
}

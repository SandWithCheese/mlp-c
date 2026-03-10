#include "inc/mlp.h"
#include "inc/neural_network.h"
#include "inc/perceptron.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// TODO
// 1. For each train data, we get n input
// 2.

int main() {
  srand(time(NULL));
  NeuralNetwork nn;
  InitInputLayer(&nn, 3);
  // nn.input_layer.perceptrons[0].output = 1;
  // nn.input_layer.perceptrons[1].output = 0;
  // nn.input_layer.perceptrons[2].output = 1;

  InitHiddenLayer(&nn);
  AddHiddenLayer(&nn, 3, RELU, UNIFORM);
  AddHiddenLayer(&nn, 2, RELU, UNIFORM);
  AddHiddenLayer(&nn, 2, RELU, UNIFORM);
  PrintNeuralNetwork(nn);
  FeedForward(&nn);
  printf("--------------------------------------\n");
  PrintNeuralNetwork(nn);
  return 0;
}

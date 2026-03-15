#include "inc/layer.h"
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
  AddHiddenLayer(&nn, 3, 3, RELU, UNIFORM);
  AddHiddenLayer(&nn, 3, 2, RELU, UNIFORM);
  AddHiddenLayer(&nn, 2, 2, RELU, UNIFORM);
  PrintNeuralNetwork(nn);
  printf("-----------------------------------------------------------\n");
  FeedForward(&nn);
  Layer mock_label_layer;
  InitLayer(&mock_label_layer, 0, 2, LABEL, ACTIVATION_NONE, DISTRIBUTION_NONE);
  mock_label_layer.perceptrons[0].output = 1;
  mock_label_layer.perceptrons[1].output = 0;
  PrintNeuralNetwork(nn);
  printf("-----------------------------------------------------------\n");
  BackPropagation(&nn, &mock_label_layer, BINARY_CTL);
  printf("-----------------------------------------------------------\n");
  PrintNeuralNetwork(nn);
  return 0;
}

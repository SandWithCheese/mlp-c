#include "../inc/neural_network.h"
#include <stdio.h>

void InitInputLayer(NeuralNetwork *nn, size_t capacity) {
  InitLayer(&(nn->input_layer), capacity, INPUT, ACTIVATION_NONE,
            DISTRIBUTION_NONE);
}

void InitHiddenLayer(NeuralNetwork *nn) {
  nn->hidden_layer.layers = NULL;
  nn->hidden_layer.capacity = 0;
  nn->hidden_layer.count = 0;
}

void AddHiddenLayer(NeuralNetwork *nn, size_t capacity,
                    enum ActivationType activation_type,
                    enum DistributionType distribution_type) {
  Layer layer;
  InitLayer(&layer, capacity, HIDDEN, activation_type, distribution_type);
  AddLayer(&(nn->hidden_layer), &layer);
}

void FeedForward(NeuralNetwork *nn) {
  Predict(&(nn->input_layer), &(nn->hidden_layer.layers[0]));
  for (int i = 1; i < nn->hidden_layer.count; i++) {
    Predict(&(nn->hidden_layer.layers[i - 1]), &(nn->hidden_layer.layers[i]));
  }
}

void BackPropagation(NeuralNetwork *nn, enum LossType loss_type,
                     Layer *label_layer) {
  size_t last_layer_idx = nn->hidden_layer.count - 1;
  Layer output_layer = nn->hidden_layer.layers[last_layer_idx];
  double loss = LossFunction(loss_type, &output_layer, label_layer);
  printf("Loss: %lf\n", loss);

  // TODO: Implement backprop
}

void PrintNeuralNetwork(NeuralNetwork nn) {
  printf("Input Layer\n");
  printf("Count: %zu, Capacity: %zu, LayerType: %d, ActivationType: %d\n\n",
         nn.input_layer.count, nn.input_layer.capacity,
         nn.input_layer.layer_type, nn.input_layer.activation_type);
  printf("Perceptrons\n");
  for (int i = 0; i < nn.input_layer.count; i++) {
    printf("Perceptron %d\n", i + 1);
    Perceptron p = nn.input_layer.perceptrons[i];
    printf("Weight: %f, Bias: %f, Output: %f\n", p.weight, p.bias, p.output);
  }
  printf("\n");

  printf("Hidden Layer\n");
  printf("Count: %zu, Capacity: %zu\n\n", nn.hidden_layer.count,
         nn.hidden_layer.capacity);
  printf("Layers\n");
  for (int i = 0; i < nn.hidden_layer.count; i++) {
    printf("Layer %d\n", i + 1);
    Layer l = nn.hidden_layer.layers[i];
    printf("Count: %zu, Capacity: %zu, LayerType: %d, ActivationType: %d\n\n",
           l.count, l.capacity, l.layer_type, l.activation_type);
    for (int j = 0; j < l.count; j++) {
      printf("Perceptron %d\n", j + 1);
      Perceptron p = l.perceptrons[j];
      printf("Weight: %f, Bias: %f, Output: %f\n", p.weight, p.bias, p.output);
    }
    printf("\n");
  }
}
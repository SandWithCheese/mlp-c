#include "../inc/layer.h"
#include <math.h>

void InitLayer(Layer *layer, size_t capacity, enum LayerType layer_type,
               enum ActivationType activation_type,
               enum DistributionType distribution_type) {
  layer->perceptrons = (Perceptron *)malloc(capacity * sizeof(Perceptron));
  layer->capacity = capacity;
  layer->count = capacity;
  layer->layer_type = layer_type;
  layer->activation_type = activation_type;
  for (int i = 0; i < layer->count; i++) {
    InitPerceptron(&(layer->perceptrons[i]), distribution_type);
  }
}

// double Predict(Layer *layer, enum ActivationType type) {
//   double sum = 0;
//   for (int i = 0; i < layer->count; i++) {
//     Perceptron p = layer->perceptrons[i * sizeof(*layer)];
//     sum += Calculate(p, )
//   }
// }

double ActivationFunction(enum ActivationType type, double output) {
  switch (type) {
  case DISTRIBUTION_NONE:
    return 0;
  case RELU:
    return fmax(0, output);
  case SIGMOID:
    return 1 / (1 + exp(-output));
  case TANH:
    return tanh(output);
  }

  return 0;
}
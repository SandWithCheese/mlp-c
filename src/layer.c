#include "../inc/layer.h"

void InitLayer(Layer *layer, size_t capacity, enum LayerType layer_type,
               enum ActivationType activation_type) {
  layer->perceptrons = (Perceptron *)malloc(capacity * sizeof(Perceptron));
  layer->capacity = capacity;
  layer->count = capacity;
  layer->layer_type = layer_type;
  layer->activation_type = activation_type;
}

// double Predict(Layer *layer, enum ActivationType type) {
//   double sum = 0;
//   for (int i = 0; i < layer->count; i++) {
//     Perceptron p = layer->perceptrons[i * sizeof(*layer)];
//     sum += Calculate(p, )
//   }
// }

double ActivationFunction(enum ActivationType type) {
  switch (type) {
  case 0:
    // Handle for ReLU
    break;
  case 1:
    // Handle for Sigmoid
    break;
  }

  return 0;
}
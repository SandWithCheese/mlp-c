#ifndef LAYER_H
#define LAYER_H

#include "perceptron.h"

#include <stdlib.h>

enum LayerType {
  INPUT,
  HIDDEN,
};

enum ActivationType {
  ACTIVATION_NONE, // Default if layer type = INPUT
  RELU,
  SIGMOID,
  TANH,
};

typedef struct {
  Perceptron *perceptrons;
  size_t count;
  size_t capacity;
  enum LayerType layer_type;
  enum ActivationType activation_type;
} Layer;

void InitLayer(Layer *layer, size_t capacity, enum LayerType layer_type,
               enum ActivationType activation_type,
               enum DistributionType distribution_type);

void Predict(Layer *previous_layer, Layer *current_layer);

double ActivationFunction(enum ActivationType type, double output);

#endif
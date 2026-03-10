#ifndef LAYER_H
#define LAYER_H

#include "perceptron.h"

#include <stdlib.h>

enum LayerType {
  INPUT,
  HIDDEN,
};

enum ActivationType {
  NONE, // Default if layer type = INPUT
  RELU,
  SIGMOID,
};

typedef struct {
  Perceptron *perceptrons;
  size_t count;
  size_t capacity;
  enum LayerType layer_type;
  enum ActivationType activation_type;
} Layer;

void InitLayer(Layer *layer, size_t capacity, enum LayerType layer_type,
               enum ActivationType activation_type);

double Predict(Layer *layer, enum ActivationType type);

double ActivationFunction(enum ActivationType type);

#endif
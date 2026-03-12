#ifndef LAYER_H
#define LAYER_H

#include "perceptron.h"

#include <stdlib.h>

enum LayerType {
  INPUT,
  HIDDEN,
  LABEL,
};

enum ActivationType {
  ACTIVATION_NONE, // Default if layer type = INPUT
  RELU,
  SIGMOID,
  TANH,
};

enum LossType { MSE, MAE, BINARY_CTL, CATEGORICAL_CTL };

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

double MeanSquaredError(Layer *output_layer, Layer *label_layer);

double MeanAbsoluteError(Layer *output_layer, Layer *label_layer);

double BinaryCrossEntropyLoss(Layer *output_layer, Layer *label_layer);

double CategoricalCrossEntropyLoss(Layer *output_layer, Layer *label_layer);

double LossFunction(enum LossType type, Layer *output_layer,
                    Layer *label_layer);

#endif
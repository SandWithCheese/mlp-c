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

void InitLayer(Layer *layer, size_t previous_layer_capacity, size_t capacity,
               enum LayerType layer_type, enum ActivationType activation_type,
               enum DistributionType distribution_type);

void Calculate(Perceptron *perceptron, Layer previous_layer);

void Predict(Layer *previous_layer, Layer *current_layer);

void ReLUFunction(Layer *layer);

void SigmoidFunction(Layer *layer);

double DSigmoidFunction(double x);

void TanhFunction(Layer *layer);

void ActivationFunction(enum ActivationType type, Layer *layer);

double DerivativeActivationFunction(enum ActivationType type, double net);

double MeanSquaredError(Layer *output_layer, Layer *label_layer);

double MeanAbsoluteError(Layer *output_layer, Layer *label_layer);

double BinaryCrossEntropyLoss(Layer *output_layer, Layer *label_layer);

double CategoricalCrossEntropyLoss(Layer *output_layer, Layer *label_layer);

double LossFunction(enum LossType type, Layer *output_layer,
                    Layer *label_layer);

double DerivativeMeanSquaredError(double y_hat, double y);

double DerivativeMeanAbsoluteError(double y_hat, double y);

double DerivativeBinaryCrossEntropyLoss(double y_hat, double y);

double DerivativeCategoricalCrossEntropyLoss(double y_hat, double y);

double DerivativeLossFunction(enum LossType type, double y_hat, double y);

#endif
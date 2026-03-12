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

void Predict(Layer *previous_layer, Layer *current_layer) {
  for (int i = 0; i < current_layer->count; i++) {
    double sum = 0;
    for (int j = 0; j < previous_layer->count; j++) {
      sum += Calculate(current_layer->perceptrons[i],
                       previous_layer->perceptrons[j].output);
    }
    current_layer->perceptrons[i].output =
        ActivationFunction(current_layer->activation_type, sum);
  }
}

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

double MeanSquaredError(Layer *output_layer, Layer *label_layer) {
  double sum = 0;
  for (int i = 0; i < label_layer->count; i++) {
    sum += pow(label_layer->perceptrons[i].output -
                   output_layer->perceptrons[i].output,
               2);
  }

  return sum / label_layer->count;
}

double MeanAbsoluteError(Layer *output_layer, Layer *label_layer) {
  double sum = 0;
  for (int i = 0; i < label_layer->count; i++) {
    sum += fabs(label_layer->perceptrons[i].output -
                output_layer->perceptrons[i].output);
  }

  return sum / label_layer->count;
}

double BinaryCrossEntropyLoss(Layer *output_layer, Layer *label_layer) {
  const double eps = 1e-15;
  double sum = 0;
  for (int i = 0; i < label_layer->count; i++) {
    double y_hat = fmax(eps, fmin(1.0 - eps, output_layer->perceptrons[i].output));
    double y = label_layer->perceptrons[i].output;
    sum += y * log(y_hat) + (1 - y) * log(1 - y_hat);
  }

  return -(sum / label_layer->count);
}

double CategoricalCrossEntropyLoss(Layer *output_layer, Layer *label_layer) {
  const double eps = 1e-15;
  double sum = 0;
  for (int i = 0; i < label_layer->count; i++) {
    double y_hat = fmax(eps, output_layer->perceptrons[i].output);
    sum += label_layer->perceptrons[i].output * log(y_hat);
  }

  return -sum;
}

double LossFunction(enum LossType type, Layer *output_layer,
                    Layer *label_layer) {
  switch (type) {
  case MSE:
    return MeanSquaredError(output_layer, label_layer);
  case MAE:
    return MeanAbsoluteError(output_layer, label_layer);
  case BINARY_CTL:
    return BinaryCrossEntropyLoss(output_layer, label_layer);
  case CATEGORICAL_CTL:
    return CategoricalCrossEntropyLoss(output_layer, label_layer);
  }

  return 0;
}

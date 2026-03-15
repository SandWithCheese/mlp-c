#include "../inc/layer.h"
#include <math.h>

void InitLayer(Layer *layer, size_t previous_layer_capacity, size_t capacity,
               enum LayerType layer_type, enum ActivationType activation_type,
               enum DistributionType distribution_type) {
  layer->perceptrons = (Perceptron *)malloc(capacity * sizeof(Perceptron));
  layer->capacity = capacity;
  layer->count = capacity;
  layer->layer_type = layer_type;
  layer->activation_type = activation_type;
  for (int i = 0; i < layer->count; i++) {
    InitPerceptron(&(layer->perceptrons[i]), previous_layer_capacity,
                   distribution_type);
  }
}

void Calculate(Perceptron *perceptron, Layer previous_layer) {
  double net = perceptron->bias;
  for (int i = 0; i < perceptron->weights.count; i++) {
    net +=
        perceptron->weights.weights[i] * previous_layer.perceptrons[i].output;
  }
  perceptron->net = net;
}

void Predict(Layer *previous_layer, Layer *current_layer) {
#pragma omp parallel for
  for (int i = 0; i < current_layer->count; i++) {
    Calculate(&(current_layer->perceptrons[i]), *previous_layer);
  }

  ActivationFunction(current_layer->activation_type, current_layer);
}

void ReLUFunction(Layer *layer) {
  for (int i = 0; i < layer->count; i++) {
    layer->perceptrons[i].output = fmax(0, layer->perceptrons[i].net);
  }
}

void SigmoidFunction(Layer *layer) {
  for (int i = 0; i < layer->count; i++) {
    layer->perceptrons[i].output = 1 / (1 + exp(-layer->perceptrons[i].net));
  }
}

double DSigmoidFunction(double x) { return 1 / (1 + exp(-x)); }

void TanhFunction(Layer *layer) {
  for (int i = 0; i < layer->count; i++) {
    layer->perceptrons[i].output = tanh(layer->perceptrons[i].net);
  }
}

void SoftmaxFunction(Layer *layer) {
  double max_net = layer->perceptrons[0].net;
  for (int j = 1; j < layer->count; j++) {
    if (layer->perceptrons[j].net > max_net)
      max_net = layer->perceptrons[j].net;
  }

  double sum = 0;
  for (int j = 0; j < layer->count; j++) {
    sum += exp(layer->perceptrons[j].net - max_net);
  }

  for (int i = 0; i < layer->count; i++) {
    layer->perceptrons[i].output =
        exp(layer->perceptrons[i].net - max_net) / sum;
  }
}

void ActivationFunction(enum ActivationType type, Layer *layer) {
  switch (type) {
  case ACTIVATION_NONE:
    return;
  case RELU:
    ReLUFunction(layer);
    return;
  case SIGMOID:
    SigmoidFunction(layer);
    return;
  case TANH:
    TanhFunction(layer);
    return;
  case SOFTMAX:
    SoftmaxFunction(layer);
    return;
  }

  return;
}

double DerivativeActivationFunction(enum ActivationType type, double net) {
  switch (type) {
  case ACTIVATION_NONE:
    return 0;
  case RELU:
    return net <= 0 ? 0 : 1;
  case SIGMOID:
    return DSigmoidFunction(net) * (1 - DSigmoidFunction(net));
  case TANH:
    return 1 - pow(tanh(net), 2);
  case SOFTMAX:
    // Assuming that softmax will only be on the output layer with Categorical
    // Cross Entropy Loss
    return 1;
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
    double y_hat =
        fmax(eps, fmin(1.0 - eps, output_layer->perceptrons[i].output));
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

double DerivativeMeanSquaredError(double y_hat, double y) {
  return (2 * (y_hat - y));
}

double DerivativeMeanAbsoluteError(double y_hat, double y) {
  double diff = y_hat - y;
  return (diff > 0) - (diff < 0);
}

double DerivativeBinaryCrossEntropyLoss(double y_hat, double y) {
  const double eps = 1e-15;
  double y_pred = fmax(eps, fmin(1 - eps, y_hat));
  return -(y / y_pred - (1 - y) / (1 - y_pred));
}

double DerivativeCategoricalCrossEntropyLoss(double y_hat, double y) {
  const double eps = 1e-15;
  double y_pred = fmax(eps, y_hat);
  return -(y / y_pred);
}

double DerivativeLossFunction(enum LossType type, double y_hat, double y) {
  switch (type) {
  case MSE:
    return DerivativeMeanSquaredError(y_hat, y);
  case MAE:
    return DerivativeMeanAbsoluteError(y_hat, y);
  case BINARY_CTL:
    return DerivativeBinaryCrossEntropyLoss(y_hat, y);
  case CATEGORICAL_CTL:
    return DerivativeCategoricalCrossEntropyLoss(y_hat, y);
  }

  return 0;
}
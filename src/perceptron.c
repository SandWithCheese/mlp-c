#include "../inc/perceptron.h"
#include <math.h>
#include <stdlib.h>

double RandomUniform(double low, double high) {
  double random_double = (double)rand() / ((double)RAND_MAX + 1.0);
  return low + (high - low) * random_double;
}

double RandomNormal(double mu, double sigma) {
  double u1 = (double)rand() / ((double)RAND_MAX + 1.0);
  double u2 = (double)rand() / ((double)RAND_MAX + 1.0);
  double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
  return mu + sigma * z;
}

double RandomHE(double mu, double n) { return RandomNormal(mu, sqrt(2 / n)); }

double RandomUniformHavier(double input, double output) {
  double x = sqrt(6 / (input + output));
  return RandomUniform(-x, x);
}

double RandomNormalHavier(double input, double output) {
  double sigma = sqrt(2 / (input + output));
  return RandomNormal(0, sigma);
}

void InitPerceptron(Perceptron *perceptron, size_t previous_layer_capacity,
                    size_t capacity, enum DistributionType type) {
  switch (type) {
  case DISTRIBUTION_NONE:
    perceptron->weights.weights = NULL;
    perceptron->weights.count = 0;
    perceptron->weights.capacity = 0;
    perceptron->bias = 0;
    perceptron->output = 0;
    perceptron->net = 0;
    perceptron->delta = 0;
    break;
  case UNIFORM:
    perceptron->weights.capacity = previous_layer_capacity;
    perceptron->weights.count = previous_layer_capacity;
    perceptron->weights.weights =
        (double *)malloc(previous_layer_capacity * sizeof(double));
    for (int i = 0; i < perceptron->weights.count; i++) {
      perceptron->weights.weights[i] = RandomUniform(-1, 1);
    }
    perceptron->bias = RandomUniform(-1, 1);
    perceptron->output = 0;
    perceptron->net = 0;
    perceptron->delta = 0;
    break;
  case NORMAL:
    perceptron->weights.capacity = previous_layer_capacity;
    perceptron->weights.count = previous_layer_capacity;
    perceptron->weights.weights =
        (double *)malloc(previous_layer_capacity * sizeof(double));
    for (int i = 0; i < perceptron->weights.count; i++) {
      perceptron->weights.weights[i] = RandomNormal(0, 1);
    }
    perceptron->bias = RandomNormal(0, 1);
    perceptron->output = 0;
    perceptron->net = 0;
    perceptron->delta = 0;
    break;
  case HE:
    perceptron->weights.capacity = previous_layer_capacity;
    perceptron->weights.count = previous_layer_capacity;
    perceptron->weights.weights =
        (double *)malloc(previous_layer_capacity * sizeof(double));
    for (int i = 0; i < perceptron->weights.count; i++) {
      perceptron->weights.weights[i] = RandomHE(0, previous_layer_capacity);
    }
    perceptron->bias = RandomHE(0, previous_layer_capacity);
    perceptron->output = 0;
    perceptron->net = 0;
    perceptron->delta = 0;
    break;
  case UNIFORM_HAVIER:
    perceptron->weights.capacity = previous_layer_capacity;
    perceptron->weights.count = previous_layer_capacity;
    perceptron->weights.weights =
        (double *)malloc(previous_layer_capacity * sizeof(double));
    for (int i = 0; i < perceptron->weights.count; i++) {
      perceptron->weights.weights[i] =
          RandomUniformHavier(previous_layer_capacity, capacity);
    }
    perceptron->bias = RandomUniformHavier(previous_layer_capacity, capacity);
    perceptron->output = 0;
    perceptron->net = 0;
    perceptron->delta = 0;
    break;
  case NORMAL_HAVIER:
    perceptron->weights.capacity = previous_layer_capacity;
    perceptron->weights.count = previous_layer_capacity;
    perceptron->weights.weights =
        (double *)malloc(previous_layer_capacity * sizeof(double));
    for (int i = 0; i < perceptron->weights.count; i++) {
      perceptron->weights.weights[i] =
          RandomNormalHavier(previous_layer_capacity, capacity);
    }
    perceptron->bias = RandomNormalHavier(previous_layer_capacity, capacity);
    perceptron->output = 0;
    perceptron->net = 0;
    perceptron->delta = 0;
    break;
  }
  return;
}

void SaveOutput(Perceptron *perceptron, double output) {
  perceptron->output = output;
}

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

void InitPerceptron(Perceptron *perceptron, size_t previous_layer_capacity,
                    enum DistributionType type) {
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
  }
  return;
}

void SaveOutput(Perceptron *perceptron, double output) {
  perceptron->output = output;
}

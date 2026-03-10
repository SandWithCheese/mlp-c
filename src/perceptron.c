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

void InitPerceptron(Perceptron *perceptron, enum DistributionType type) {
  switch (type) {
  case DISTRIBUTION_NONE:
    break;
  case UNIFORM:
    perceptron->weight = RandomUniform(-1, 1);
    perceptron->bias = RandomUniform(-1, 1);
    perceptron->output = 0;
    break;
  case NORMAL:
    perceptron->weight = RandomNormal(0, 1);
    perceptron->bias = RandomNormal(0, 1);
    perceptron->output = 0;
    break;
  }
  return;
}

double Calculate(Perceptron perceptron, double value) {
  return perceptron.weight * value + perceptron.bias;
}

void SaveOutput(Perceptron *perceptron, double output) {
  perceptron->output = output;
}

#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <stdlib.h>
typedef struct {
  double *weights;
  size_t count;
  size_t capacity;
} Weights;

typedef struct {
  Weights weights;
  double bias;
  double output;
  double net;
  double delta;
} Perceptron;

enum DistributionType {
  DISTRIBUTION_NONE, // Default if layer type = INPUT
  UNIFORM,
  NORMAL,
  HE,
  UNIFORM_HAVIER,
  NORMAL_HAVIER,
};

double RandomUniform(double low, double high);

double RandomNormal(double mu, double sigma);

double RandomHE(double mu, double n);

double RandomUniformHavier(double input, double output);

double RandomNormalHavier(double input, double output);

void InitPerceptron(Perceptron *perceptron, size_t previous_layer_capacity,
                    size_t capacity, enum DistributionType type);

void SaveOutput(Perceptron *perceptron, double output);

#endif
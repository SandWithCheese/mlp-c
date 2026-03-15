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
};

double RandomUniform(double low, double high);

double RandomNormal(double mu, double sigma);

void InitPerceptron(Perceptron *perceptron, size_t previous_layer_capacity,
                    enum DistributionType type);

void SaveOutput(Perceptron *perceptron, double output);

#endif
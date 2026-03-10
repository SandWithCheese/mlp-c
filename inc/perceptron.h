#ifndef PERCEPTRON_H
#define PERCEPTRON_H

typedef struct {
  double weight;
  double bias;
  double output;
} Perceptron;

enum DistributionType {
  DISTRIBUTION_NONE, // Default if layer type = INPUT
  UNIFORM,
  NORMAL,
};

double RandomUniform(double low, double high);

double RandomNormal(double mu, double sigma);

void InitPerceptron(Perceptron *perceptron, enum DistributionType type);

double Calculate(Perceptron perceptron, double value);

void SaveOutput(Perceptron *perceptron, double output);

#endif
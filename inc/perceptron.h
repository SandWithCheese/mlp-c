#ifndef PERCEPTRON_H
#define PERCEPTRON_H

typedef struct {
  double weight;
  double bias;
  double output;
} Perceptron;

double Calculate(Perceptron perceptron, double value);

void SaveOutput(Perceptron *perceptron, double output);

#endif
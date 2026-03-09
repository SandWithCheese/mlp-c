#include "../inc/perceptron.h"

double Calculate(Perceptron perceptron, double value) {
  return perceptron.weight * value + perceptron.bias;
}

void SaveOutput(Perceptron *perceptron, double output) {
  perceptron->output = output;
}

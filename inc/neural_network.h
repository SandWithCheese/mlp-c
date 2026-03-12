#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "hidden_layer.h"
#include "layer.h"
#include <stdlib.h>

typedef struct {
  Layer input_layer;
  HiddenLayer hidden_layer;
} NeuralNetwork;

void InitInputLayer(NeuralNetwork *nn, size_t capacity);

void InitHiddenLayer(NeuralNetwork *nn);

void AddHiddenLayer(NeuralNetwork *nn, size_t capacity,
                    enum ActivationType activation_type,
                    enum DistributionType distribution_type);

void FeedForward(NeuralNetwork *nn);

void BackPropagation(NeuralNetwork *nn, enum LossType loss_type,
                     Layer *label_layer);

void PrintNeuralNetwork(NeuralNetwork nn);

#endif
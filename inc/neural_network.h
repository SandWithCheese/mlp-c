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

void AddHiddenLayer(NeuralNetwork *nn, size_t previous_layer_capacity,
                    size_t capacity, enum ActivationType activation_type,
                    enum DistributionType distribution_type);

void FeedForward(NeuralNetwork *nn);

double CalculateLoss(NeuralNetwork *nn, Layer *label_layer,
                     enum LossType loss_type);

void BackPropagation(NeuralNetwork *nn, Layer *label_layer,
                     enum LossType loss_type, double learning_rate);

void PrintProgress(size_t count, size_t max);

void Train(NeuralNetwork *nn, double train_data[][784], int *train_label,
           size_t num_samples, double learning_rate, size_t epochs,
           enum LossType loss_type);

void SaveNeuralNetwork(NeuralNetwork *nn, const char *filename);

void LoadNeuralNetwork(NeuralNetwork *nn, const char *filename);

void PrintNeuralNetwork(NeuralNetwork nn);

#endif
#include "../inc/neural_network.h"
#include "../lib/tpl.h"
#include <stdio.h>

void InitInputLayer(NeuralNetwork *nn, size_t capacity) {
  InitLayer(&(nn->input_layer), 0, capacity, INPUT, ACTIVATION_NONE,
            DISTRIBUTION_NONE);
}

void InitHiddenLayer(NeuralNetwork *nn) {
  nn->hidden_layer.layers = NULL;
  nn->hidden_layer.capacity = 0;
  nn->hidden_layer.count = 0;
}

void AddHiddenLayer(NeuralNetwork *nn, size_t previous_layer_capacity,
                    size_t capacity, enum ActivationType activation_type,
                    enum DistributionType distribution_type) {
  Layer layer;
  InitLayer(&layer, previous_layer_capacity, capacity, HIDDEN, activation_type,
            distribution_type);
  AddLayer(&(nn->hidden_layer), &layer);
}

void FeedForward(NeuralNetwork *nn) {
  Predict(&(nn->input_layer), &(nn->hidden_layer.layers[0]));
  for (int i = 1; i < nn->hidden_layer.count; i++) {
    Predict(&(nn->hidden_layer.layers[i - 1]), &(nn->hidden_layer.layers[i]));
  }
}

double CalculateLoss(NeuralNetwork *nn, Layer *label_layer,
                     enum LossType loss_type) {
  size_t last_layer_idx = nn->hidden_layer.count - 1;
  Layer *output_layer = &nn->hidden_layer.layers[last_layer_idx];
  return LossFunction(loss_type, output_layer, label_layer);
}

void BackPropagation(NeuralNetwork *nn, Layer *label_layer,
                     enum LossType loss_type, double learning_rate) {
  size_t last_layer_idx = nn->hidden_layer.count - 1;
  Layer *output_layer = &nn->hidden_layer.layers[last_layer_idx];

  // Calculate delta for output layer
  for (int i = 0; i < output_layer->count; i++) {
    if (output_layer->activation_type == SOFTMAX &&
        loss_type == CATEGORICAL_CTL) {
      output_layer->perceptrons[i].delta = output_layer->perceptrons[i].output -
                                           label_layer->perceptrons[i].output;
    } else {
      output_layer->perceptrons[i].delta =
          DerivativeLossFunction(loss_type, output_layer->perceptrons[i].output,
                                 label_layer->perceptrons[i].output) *
          DerivativeActivationFunction(output_layer->activation_type,
                                       output_layer->perceptrons[i].net);
    }
  }

  // Calculate delta for all hidden layers
  for (int i = last_layer_idx - 1; i >= 0; i--) {
    Layer *current_layer = &nn->hidden_layer.layers[i];
#pragma omp parallel for
    for (int j = 0; j < current_layer->count; j++) {
      double delta = 0;
      Layer next_layer = nn->hidden_layer.layers[i + 1];
      for (int k = 0; k < next_layer.count; k++) {
        Perceptron p = next_layer.perceptrons[k];
        delta += p.delta * p.weights.weights[j];
      }

      current_layer->perceptrons[j].delta =
          delta *
          DerivativeActivationFunction(current_layer->activation_type,
                                       current_layer->perceptrons[j].net);
    }
  }

  // Update weights for first hidden layer (previous = input layer)
  Layer *input_layer = &nn->input_layer;
  Layer *first_hidden_layer = &nn->hidden_layer.layers[0];
#pragma omp parallel for
  for (int i = 0; i < first_hidden_layer->count; i++) {
    Perceptron *current_perceptron = &first_hidden_layer->perceptrons[i];
    for (int j = 0; j < current_perceptron->weights.count; j++) {
      current_perceptron->weights.weights[j] -=
          learning_rate * current_perceptron->delta *
          input_layer->perceptrons[j].output;
    }
    current_perceptron->bias -= learning_rate * current_perceptron->delta;
  }

  // Update weights for the rest of hidden layers (previous = preceding hidden
  // layer)
  for (int i = 1; i < nn->hidden_layer.count; i++) {
    Layer *current_layer = &nn->hidden_layer.layers[i];
    Layer *previous_layer = &nn->hidden_layer.layers[i - 1];
#pragma omp parallel for
    for (int j = 0; j < current_layer->count; j++) {
      Perceptron *current_perceptron = &current_layer->perceptrons[j];
      for (int k = 0; k < current_perceptron->weights.count; k++) {
        current_perceptron->weights.weights[k] -=
            learning_rate * current_perceptron->delta *
            previous_layer->perceptrons[k].output;
      }
      current_perceptron->bias -= learning_rate * current_perceptron->delta;
    }
  }
}

void PrintProgress(size_t count, size_t max) {
  const int bar_width = 50;

  float progress = (float)count / max;
  int bar_length = progress * bar_width;

  printf("\rProgress: [");
  for (int i = 0; i < bar_length; ++i) {
    printf("#");
  }
  for (int i = bar_length; i < bar_width; ++i) {
    printf(" ");
  }
  printf("] %.2f%%", progress * 100);

  fflush(stdout);
}

void Train(NeuralNetwork *nn, double train_data[][784], int *train_label,
           size_t num_samples, double learning_rate, size_t epochs,
           enum LossType loss_type) {
  size_t last_layer_idx = nn->hidden_layer.count - 1;
  Layer *output_layer = &nn->hidden_layer.layers[last_layer_idx];

  Layer label_layer;
  InitLayer(&label_layer, 0, output_layer->count, LABEL, ACTIVATION_NONE,
            DISTRIBUTION_NONE);

  for (int i = 0; i < epochs; i++) {
    double error_sum = 0;
    printf("Epoch %d\n", i + 1);

    for (int j = 0; j < num_samples; j++) {
      if (j % 1000 == 0) {
        PrintProgress(j, num_samples);
      }
      for (int k = 0; k < nn->input_layer.count; k++) {
        nn->input_layer.perceptrons[k].output = train_data[j][k];
      }

      for (int k = 0; k < output_layer->count; k++) {
        label_layer.perceptrons[k].output = (k == train_label[j]) ? 1.0 : 0.0;
      }

      FeedForward(nn);

      double loss = CalculateLoss(nn, &label_layer, loss_type);
      error_sum += loss;

      BackPropagation(nn, &label_layer, loss_type, learning_rate);
    }

    printf("\nLoss: %lf\n", error_sum);

    SaveNeuralNetwork(nn, "./weights/best.tpl");
  }
}

void SaveNeuralNetwork(NeuralNetwork *nn, const char *filename) {
  double w_val, bias, output, net, delta;
  int layer_type, activation_type;

  tpl_node *tn = tpl_map("A(A(A(f)ffff)ii)", &w_val, &bias, &output, &net,
                         &delta, &layer_type, &activation_type);

  for (size_t i = 0; i < nn->hidden_layer.count; i++) {
    Layer *l = &nn->hidden_layer.layers[i];
    layer_type = l->layer_type;
    activation_type = l->activation_type;

    for (size_t j = 0; j < l->count; j++) {
      Perceptron *p = &l->perceptrons[j];
      bias = p->bias;
      output = p->output;
      net = p->net;
      delta = p->delta;

      for (size_t k = 0; k < p->weights.count; k++) {
        w_val = p->weights.weights[k];
        tpl_pack(tn, 3);
      }
      tpl_pack(tn, 2);
    }
    tpl_pack(tn, 1);
  }

  tpl_dump(tn, TPL_FILE, filename);
  tpl_free(tn);
}

void LoadNeuralNetwork(NeuralNetwork *nn, const char *filename) {
  double w_val, bias, output, net, delta;
  int layer_type, activation_type;

  tpl_node *tn = tpl_map("A(A(A(f)ffff)ii)", &w_val, &bias, &output, &net,
                         &delta, &layer_type, &activation_type);

  if (tpl_load(tn, TPL_FILE, filename) != 0) {
    fprintf(stderr, "Failed to load TPL file.\n");
    tpl_free(tn);
    return;
  }

  int layer_count = tpl_Alen(tn, 1);
  nn->hidden_layer.layers = malloc(sizeof(Layer) * layer_count);
  nn->hidden_layer.count = layer_count;
  nn->hidden_layer.capacity = layer_count;

  int l_idx = 0;
  while (tpl_unpack(tn, 1) > 0) {
    Layer *current_l = &nn->hidden_layer.layers[l_idx++];
    current_l->layer_type = layer_type;
    current_l->activation_type = activation_type;

    int perc_count = tpl_Alen(tn, 2);
    current_l->perceptrons = malloc(sizeof(Perceptron) * perc_count);
    current_l->count = perc_count;
    current_l->capacity = perc_count;

    int p_idx = 0;
    while (tpl_unpack(tn, 2) > 0) {
      Perceptron *current_p = &current_l->perceptrons[p_idx++];
      current_p->bias = bias;
      current_p->output = output;
      current_p->net = net;
      current_p->delta = delta;

      int weight_count = tpl_Alen(tn, 3);
      current_p->weights.weights = malloc(sizeof(double) * weight_count);
      current_p->weights.count = weight_count;
      current_p->weights.capacity = weight_count;

      int w_idx = 0;
      while (tpl_unpack(tn, 3) > 0) {
        current_p->weights.weights[w_idx++] = w_val;
      }
    }
  }

  tpl_free(tn);
}

void PrintNeuralNetwork(NeuralNetwork nn) {
  printf("Input Layer\n");
  printf("Count: %zu, Capacity: %zu, LayerType: %d, ActivationType: %d\n\n",
         nn.input_layer.count, nn.input_layer.capacity,
         nn.input_layer.layer_type, nn.input_layer.activation_type);
  printf("Perceptrons\n");
  for (int i = 0; i < nn.input_layer.count; i++) {
    printf("Perceptron %d\n", i + 1);
    Perceptron p = nn.input_layer.perceptrons[i];
    printf("Weights\n");
    for (int j = 0; j < p.weights.count; j++) {
      printf("Weight: %lf\n", p.weights.weights[j]);
    }
    printf("Bias: %lf, Output: %lf, Net: %lf, Delta: %lf\n", p.bias, p.output,
           p.net, p.delta);
  }
  printf("\n");

  printf("Hidden Layer\n");
  printf("Count: %zu, Capacity: %zu\n\n", nn.hidden_layer.count,
         nn.hidden_layer.capacity);
  printf("Layers\n");
  for (int i = 0; i < nn.hidden_layer.count; i++) {
    printf("Layer %d\n", i + 1);
    Layer l = nn.hidden_layer.layers[i];
    printf("Count: %zu, Capacity: %zu, LayerType: %d, ActivationType: %d\n\n",
           l.count, l.capacity, l.layer_type, l.activation_type);
    for (int j = 0; j < l.count; j++) {
      printf("Perceptron %d\n", j + 1);
      Perceptron p = l.perceptrons[j];
      printf("Weights\n");
      for (int k = 0; k < p.weights.count; k++) {
        printf("Weight: %lf\n", p.weights.weights[k]);
      }
      printf("Bias: %lf, Output: %lf, Net: %lf, Delta: %lf\n", p.bias, p.output,
             p.net, p.delta);
    }
    printf("\n");
  }
}
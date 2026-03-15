#include "inc/layer.h"
#include "inc/mlp.h"
#include "inc/neural_network.h"
#include "inc/perceptron.h"
#include "lib/mnist.h"
#include <stdlib.h>
#include <time.h>

int main() {
  srand(time(NULL));

  load_mnist();

  NeuralNetwork nn;
  InitInputLayer(&nn, 784);
  InitHiddenLayer(&nn);
  AddHiddenLayer(&nn, 784, 512, RELU, HE);
  AddHiddenLayer(&nn, 512, 256, RELU, HE);
  AddHiddenLayer(&nn, 256, 10, SOFTMAX, UNIFORM_HAVIER);

  Train(&nn, train_image, train_label, NUM_TRAIN, 0.001, 10, 0.2,
        CATEGORICAL_CTL);
  return 0;
}

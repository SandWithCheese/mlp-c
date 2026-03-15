#include "inc/layer.h"
#include "inc/mlp.h"
#include "inc/neural_network.h"
#include "inc/perceptron.h"
#include "lib/mnist.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("usage: ./main {train,infer}");
    return 1;
  }
  srand(time(NULL));

  load_mnist();

  NeuralNetwork nn;

  if (strncmp(argv[1], "train", 5) == 0) {
    InitInputLayer(&nn, 784);
    InitHiddenLayer(&nn);
    AddHiddenLayer(&nn, 784, 512, RELU, HE);
    AddHiddenLayer(&nn, 512, 256, RELU, HE);
    AddHiddenLayer(&nn, 256, 10, SOFTMAX, UNIFORM_HAVIER);

    Train(&nn, train_image, train_label, NUM_TRAIN, 0.001, 20, 0.2,
          CATEGORICAL_CTL);
  } else {
    InitInputLayer(&nn, 784);
    LoadNeuralNetwork(&nn, "./weights/best.tpl");
    Inference(&nn, test_image, test_label, NUM_TEST);
  }
  return 0;
}

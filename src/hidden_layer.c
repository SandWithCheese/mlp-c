#include "../inc/hidden_layer.h"
#include <stdlib.h>

void AddLayer(HiddenLayer *hidden_layer, Layer *layer) {
  if (hidden_layer->count >= hidden_layer->capacity) {
    hidden_layer->capacity += 1;
    hidden_layer->layers =
        realloc(hidden_layer->layers,
                hidden_layer->capacity * sizeof(*(hidden_layer->layers)));
  }
  hidden_layer->layers[hidden_layer->count++] = *layer;
}
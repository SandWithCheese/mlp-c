#ifndef HIDDEN_LAYER_H
#define HIDDEN_LAYER_H

#include "layer.h"

typedef struct {
  Layer *layers;
  size_t count;
  size_t capacity;
} HiddenLayer;

void AddLayer(HiddenLayer *hidden_layer, Layer *layer);

#endif
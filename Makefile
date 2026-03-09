# CONFIGS
COMPILER = clang
MAIN = ./main.c
FLAGS = -Wall -Werror -O3
OUTPUT = ./main
IMPL = ./src/header.c ./src/hidden_layer.c ./src/layer.c ./src/perceptron.c ./src/neural_network.c

compile:
	$(COMPILER) $(MAIN) $(IMPL) $(FLAGS) -o $(OUTPUT) 

run:
	$(OUTPUT)
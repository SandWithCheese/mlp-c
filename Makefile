# CONFIGS
COMPILER = clang
MAIN = ./main.c
FLAGS = -Wall -Werror -O3 -lm
OUTPUT = ./main
IMPL = ./src/hidden_layer.c ./src/layer.c ./src/perceptron.c ./src/neural_network.c

all: compile run

compile:
	$(COMPILER) $(MAIN) $(IMPL) $(FLAGS) -o $(OUTPUT)

run:
	$(OUTPUT)

clean:
	rm -f $(OUTPUT)
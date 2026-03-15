# CONFIGS
COMPILER = clang
MAIN = ./main.c
FLAGS = -Wall -Werror -O3 -lm -fopenmp
OUTPUT = ./main
IMPL = ./src/hidden_layer.c ./src/layer.c ./src/perceptron.c ./src/neural_network.c ./lib/tpl.c
NUM_THREADS = 16

all: compile run

compile:
	$(COMPILER) $(MAIN) $(IMPL) $(FLAGS) -o $(OUTPUT)

run:
	OMP_NUM_THREADS=$(NUM_THREADS) $(OUTPUT)

clean:
	rm -f $(OUTPUT)
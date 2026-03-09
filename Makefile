# CONFIGS
COMPILER = clang
MAIN = ./main.c
FLAGS = -Wall -Werror -O3
OUTPUT = ./main
IMPL = ./src/header.c

compile:
	$(COMPILER) $(MAIN) $(IMPL) $(FLAGS) -o $(OUTPUT) 

run:
	$(OUTPUT)
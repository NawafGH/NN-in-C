CC = gcc
CFLAGS = -Wall -Wextra -O2 -lm

SRC = main.c neuralnet.c data.c
OBJ = $(SRC:.c=.o)
TARGET = mnist_model

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJ)

clean:
	rm -f $(OBJ) $(TARGET)

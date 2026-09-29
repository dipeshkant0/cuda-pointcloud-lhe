CC = nvcc
CFLAGS = -O3 -arch=sm_35 -Xcompiler -fopenmp
TARGET = a2
SRC = main.cpp knn.cu approx_knn.cu k_mean.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET) *.txt

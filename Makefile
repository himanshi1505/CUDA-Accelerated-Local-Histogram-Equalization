NVCC     = nvcc 
TARGET   = a2

all: $(TARGET)

$(TARGET): main.cu
	$(NVCC) main.cu -arch=sm_35 -Xcompiler -fopenmp -Xcompiler -static-libstdc++ -O3 -o $(TARGET)

run: $(TARGET)
	./$(TARGET) input.txt

clean:
	rm -f $(TARGET) knn.txt approx_knn.txt kmeans.txt
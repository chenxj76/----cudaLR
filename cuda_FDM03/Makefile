CUDA_LIB :=/usr/local/cuda/lib64 -lcuda -lcudart

All: A B C
	g++ FDM03cuda.o memory.o forward.o -o test.run -L $(CUDA_LIB)
A:
	nvcc -arch=sm_35 memory.cu -c
B:
	nvcc -arch=sm_35 forward.cu -c
C:
	g++ FDM03cuda.cpp -c
clean:
	rm *.o *.txt *.run

CUDA_LIB :=/usr/local/cuda/lib64 -lcuda -lcudart

All: A B C
	g++ FDMLR91.o memory.o forward.o -o test.run -L $(CUDA_LIB)
A:
	nvcc memory.cu -c
B:
	nvcc forward.cu -c
C:
	g++ FDMLR91.cpp -c
clean:
	rm *.o *.txt *.run

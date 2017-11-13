CUDA_LIB :=/usr/local/cuda/lib64 -lcuda -lcudart

All: A B C D
	g++ FDMLRI03.o memory.o ion.o voltage.o -O3 -o test.run -L $(CUDA_LIB)
A:
	nvcc -arch=sm_61  memory.cu -c
B:
	nvcc -arch=sm_61  ion.cu -c
C:
	nvcc -arch=sm_61  voltage.cu -c
D:	
	g++ FDMLRI03.cpp -c
clean:
	rm *.o *.txt *.run

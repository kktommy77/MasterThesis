NVCC = /usr/local/cuda-12.1/bin/nvcc
INCFLAGS = -I /usr/local/cuda-12.1/include
CUDAFLAGS = -shared -O2 -gencode=arch=compute_80,code=sm_80 -std=c++14
CUDAFLAGSADD = --compiler-options "-fPIC"
 
all: libcompression_kernel.so
 
libcompression_kernel.so: compression.cu
	$(NVCC) $(INCFLAGS) -o $@ -c $^ $(CUDAFLAGS) $(CUDAFLAGSADD)

clean:
	rm -f ./*.so*
	

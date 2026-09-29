CXX = g++
CXXFLAGS = -std=c++17 -O3 -fopenmp

NVCC = nvcc
CUDA_FLAGS = -O3 -arch=sm_35 -Xcompiler -fopenmp

TARGET_OMP = omp_lhe
TARGET_CUDA = a2/a2

all: $(TARGET_OMP)
	@if command -v $(NVCC) >/dev/null 2>&1; then \
		$(MAKE) cuda; \
	else \
		echo "nvcc not found in PATH; skipping CUDA build (run on CUDA-enabled machine)."; \
	fi

omp: $(TARGET_OMP)

$(TARGET_OMP): OMP/main.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET_OMP) OMP/main.cpp

cuda:
	$(MAKE) -C a2

clean:
	rm -f $(TARGET_OMP)
	$(MAKE) -C a2 clean || true

.PHONY: all omp cuda clean

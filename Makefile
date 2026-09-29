CXX = g++
CXXFLAGS = -std=c++17 -O3 -fopenmp

NVCC = nvcc

TARGET_OMP = omp_lhe

all: $(TARGET_OMP)
	@if command -v $(NVCC) >/dev/null 2>&1; then \
		$(MAKE) cuda; \
	else \
		echo "nvcc not found; skipping CUDA build (run on CUDA-enabled machine)."; \
	fi

omp: $(TARGET_OMP)

$(TARGET_OMP): src/omp/main.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET_OMP) src/omp/main.cpp

cuda:
	$(MAKE) -C src/cuda

clean:
	rm -f $(TARGET_OMP) *.txt
	$(MAKE) -C src/cuda clean || true

.PHONY: all omp cuda clean

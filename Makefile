CC = gcc
CFLAGS = -O3 -march=native -Wall -fno-tree-vectorize
AVX_FLAGS = -mavx512f -mavx512dq
FMA_FLAGS = -mfma
OMP_FLAGS = -fopenmp

TARGETS = medir benchmark_avx benchmark_unroll benchmark_fma

.PHONY: all clean run

all: $(TARGETS)

medir: medir.c
	$(CC) $(CFLAGS) $< -o $@

benchmark_avx: benchmark_avx.c
	$(CC) $(CFLAGS) $(AVX_FLAGS) $< -o $@

benchmark_unroll: benchmark_unroll.c
	$(CC) $(CFLAGS) $(AVX_FLAGS) $< -o $@

benchmark_fma: benchmark_fma.c
	$(CC) $(CFLAGS) $(AVX_FLAGS) $(FMA_FLAGS) $(OMP_FLAGS) $< -o $@

run: all
	@echo "=== 1. Medição Simples ==="
	@./medir
	@echo "\n=== 2. Benchmark AVX-512 ==="
	@./benchmark_avx
	@echo "\n=== 3. Benchmark Unrolling (ILP) ==="
	@./benchmark_unroll
	@echo "\n=== 4. Benchmark FMA Matrizes (com OpenMP) ==="
	@./benchmark_fma

clean:
	rm -f $(TARGETS)
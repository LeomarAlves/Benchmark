#define _POSIX_C_SOURCE 200112L
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <x86intrin.h>
#include <omp.h>

#define N 1024

static inline unsigned long long rdtsc_fence() {
    _mm_lfence();
    return __rdtsc();
}

int main() {
    size_t bytes = N * N * sizeof(float);
    float *A        = (float *)aligned_alloc(64, bytes);
    float *B        = (float *)aligned_alloc(64, bytes);
    float *C_seq    = (float *)aligned_alloc(64, bytes);
    float *C_avx    = (float *)aligned_alloc(64, bytes);
    float *C_omp    = (float *)aligned_alloc(64, bytes);

    if (!A || !B || !C_seq || !C_avx || !C_omp) {
        perror("Erro de alocação");
        return 1;
    }

    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0f; B[i] = 2.0f;
        C_seq[i] = 0.0f; C_avx[i] = 0.0f; C_omp[i] = 0.0f;
    }

    double total_flops = 2.0 * (double)N * (double)N * (double)N;

    // 1. ESCALAR PADRÃO (1 thread)
    unsigned long long t0 = rdtsc_fence();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float soma = 0.0f;
            for (int k = 0; k < N; k++) {
                soma += A[i * N + k] * B[k * N + j];
            }
            C_seq[i * N + j] = soma;
        }
    }
    unsigned long long ciclos_seq = rdtsc_fence() - t0;

    // 2. AVX-512 + FMA (1 thread)
    unsigned long long t1 = rdtsc_fence();
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            __m512 a_ik = _mm512_set1_ps(A[i * N + k]);
            for (int j = 0; j < N; j += 16) {
                __m512 c_ij = _mm512_load_ps(&C_avx[i * N + j]);
                __m512 b_kj = _mm512_load_ps(&B[k * N + j]);
                c_ij = _mm512_fmadd_ps(a_ik, b_kj, c_ij);
                _mm512_store_ps(&C_avx[i * N + j], c_ij);
            }
        }
    }
    unsigned long long ciclos_avx = rdtsc_fence() - t1;

    // 3. AVX-512 + FMA + OpenMP (Multi-thread nos 8 threads lógicos)
    int max_threads = omp_get_max_threads();
    unsigned long long t2 = rdtsc_fence();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            __m512 a_ik = _mm512_set1_ps(A[i * N + k]);
            for (int j = 0; j < N; j += 16) {
                __m512 c_ij = _mm512_load_ps(&C_omp[i * N + j]);
                __m512 b_kj = _mm512_load_ps(&B[k * N + j]);
                c_ij = _mm512_fmadd_ps(a_ik, b_kj, c_ij);
                _mm512_store_ps(&C_omp[i * N + j], c_ij);
            }
        }
    }
    unsigned long long ciclos_omp = rdtsc_fence() - t2;

    // EXIBIÇÃO
    printf("Matrizes: %dx%d | Total FLOPs: %.0f | Threads OpenMP: %d\n\n", N, N, total_flops, max_threads);
    printf("Estratégia               | Ciclos Totais | FLOPs/Ciclo | Speedup\n");
    printf("-------------------------|---------------|-------------|--------\n");
    printf("1. Escalar (1 thread)    | %13llu | %11.2f | 1.00x\n", 
           ciclos_seq, total_flops / ciclos_seq);
    printf("2. AVX-512 FMA (1 thread)| %13llu | %11.2f | %.2fx\n", 
           ciclos_avx, total_flops / ciclos_avx, (double)ciclos_seq / ciclos_avx);
    printf("3. AVX-512 FMA + OpenMP  | %13llu | %11.2f | %.2fx\n", 
           ciclos_omp, total_flops / ciclos_omp, (double)ciclos_seq / ciclos_omp);

    free(A); free(B); free(C_seq); free(C_avx); free(C_omp);
    return 0;
}
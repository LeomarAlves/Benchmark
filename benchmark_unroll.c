#define _POSIX_C_SOURCE 200112L
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <x86intrin.h>

#define N 1048576 // 2^20 elementos (múltiplo exato de 64)

static inline unsigned long long rdtsc_fence() {
    _mm_lfence();
    return __rdtsc();
}

int main() {
    float *dados = (float *)aligned_alloc(64, N * sizeof(float));
    if (!dados) {
        perror("Erro de alocação");
        return 1;
    }

    for (int i = 0; i < N; i++) {
        dados[i] = 1.0f;
    }

    // 1. ESCALAR
    volatile float soma_seq = 0.0f;
    unsigned long long t0 = rdtsc_fence();
    for (int i = 0; i < N; i++) {
        soma_seq += dados[i];
    }
    unsigned long long t1 = rdtsc_fence();
    unsigned long long ciclos_seq = t1 - t0;

    // 2. AVX-512 (1 ACUMULADOR - 16 floats/iteração)
    __m512 acc = _mm512_setzero_ps();
    unsigned long long t2 = rdtsc_fence();
    for (int i = 0; i < N; i += 16) {
        acc = _mm512_add_ps(acc, _mm512_load_ps(&dados[i]));
    }
    float soma_avx1 = _mm512_reduce_add_ps(acc);
    unsigned long long t3 = rdtsc_fence();
    unsigned long long ciclos_avx1 = t3 - t2;

    // 3. AVX-512 DESENROLADO (4 ACUMULADORES - 64 floats/iteração)
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();

    unsigned long long t4 = rdtsc_fence();
    for (int i = 0; i < N; i += 64) {
        acc0 = _mm512_add_ps(acc0, _mm512_load_ps(&dados[i]));
        acc1 = _mm512_add_ps(acc1, _mm512_load_ps(&dados[i + 16]));
        acc2 = _mm512_add_ps(acc2, _mm512_load_ps(&dados[i + 32]));
        acc3 = _mm512_add_ps(acc3, _mm512_load_ps(&dados[i + 48]));
    }
    // Redução em árvore dos 4 registradores
    acc0 = _mm512_add_ps(acc0, acc1);
    acc2 = _mm512_add_ps(acc2, acc3);
    acc0 = _mm512_add_ps(acc0, acc2);
    float soma_avx4 = _mm512_reduce_add_ps(acc0);

    unsigned long long t5 = rdtsc_fence();
    unsigned long long ciclos_avx4 = t5 - t4;

    // EXIBIÇÃO
    printf("Elementos: %d\n\n", N);
    printf("Estratégia           | Resultado  | Ciclos Totais | Ciclos/elem | Speedup\n");
    printf("---------------------|------------|---------------|-------------|--------\n");
    printf("1. Escalar           | %10.0f | %13llu | %11.2f | 1.00x\n", 
           soma_seq, ciclos_seq, (double)ciclos_seq / N);
    printf("2. AVX-512 (1x Acc)  | %10.0f | %13llu | %11.2f | %.2fx\n", 
           soma_avx1, ciclos_avx1, (double)ciclos_avx1 / N, (double)ciclos_seq / ciclos_avx1);
    printf("3. AVX-512 (4x Acc)  | %10.0f | %13llu | %11.2f | %.2fx\n", 
           soma_avx4, ciclos_avx4, (double)ciclos_avx4 / N, (double)ciclos_seq / ciclos_avx4);

    free(dados);
    return 0;
}
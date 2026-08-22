#define _POSIX_C_SOURCE 200112L
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <x86intrin.h>

#define N 1048576 // 2^20 elementos (~1 milhão, múltiplo exato de 16)

// Barreira de serialização para evitar reordenação fora de ordem (Out-of-Order)
static inline unsigned long long rdtsc_fence() {
    _mm_lfence();
    return __rdtsc();
}

int main() {
    // Aloca memória alinhada em 64 bytes (tamanho exato de um registrador ZMM de 512 bits)
    float *dados = (float *)aligned_alloc(64, N * sizeof(float));
    if (!dados) {
        perror("Erro de alocação");
        return 1;
    }

    for (int i = 0; i < N; i++) {
        dados[i] = 1.0f;
    }

    // --- 1. SOMA SEQUENCIAL ESCALAR ---
    volatile float soma_seq = 0.0f;
    unsigned long long t0 = rdtsc_fence();

    for (int i = 0; i < N; i++) {
        soma_seq += dados[i];
    }

    unsigned long long t1 = rdtsc_fence();
    unsigned long long ciclos_seq = t1 - t0;

    // --- 2. SOMA VETORIZADA AVX-512 ---
    __m512 vec_soma = _mm512_setzero_ps(); // Zera o registrador ZMM
    unsigned long long t2 = rdtsc_fence();

    for (int i = 0; i < N; i += 16) {
        __m512 v = _mm512_load_ps(&dados[i]);       // Carrega 16 floats (512 bits) de uma vez
        vec_soma = _mm512_add_ps(vec_soma, v);     // Soma 16 floats em paralelo em 1 ciclo
    }
    float soma_avx = _mm512_reduce_add_ps(vec_soma); // Redução horizontal final (soma as 16 pistas)

    unsigned long long t3 = rdtsc_fence();
    unsigned long long ciclos_avx = t3 - t2;

    // --- RESULTADOS ---
    printf("Elementos processados: %d\n", N);
    printf("Escalar  | Resultado: %.0f | Ciclos: %10llu | Ciclos/elem: %.2f\n", 
           soma_seq, ciclos_seq, (double)ciclos_seq / N);
    printf("AVX-512  | Resultado: %.0f | Ciclos: %10llu | Ciclos/elem: %.2f\n", 
           soma_avx, ciclos_avx, (double)ciclos_avx / N);
    printf("Speedup  | %.2fx mais rápido\n", (double)ciclos_seq / ciclos_avx);

    free(dados);
    return 0;
}
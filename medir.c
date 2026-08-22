#include <stdio.h>
#include <x86intrin.h>

int main() {
    unsigned long long inicio, fim;

    // Captura os ciclos antes da execução
    inicio = __rdtsc();

    // Trecho de código que você deseja medir
    for (volatile int i = 0; i < 1000000; i++);

    // Captura os ciclos após a execução
    fim = __rdtsc();

    printf("Ciclos de clock decorridos: %llu\n", fim - inicio);
    return 0;
}
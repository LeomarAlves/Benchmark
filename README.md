# 🚀 Suite de Benchmarks SIMD & Otimização de Baixo Nível (AVX-512 / FMA / OpenMP)

Este repositório contém um conjunto de benchmarks de alta performance em **C** voltados para análise de desempenho, computação vetorial (**SIMD** via **AVX-512** e **FMA**), paralelismo em nível de instrução (**ILP**), otimização de localidade de cache e paralelismo *multithreading* com **OpenMP** em arquiteturas x86-64.

---

## 📁 Estrutura do Repositório

| Arquivo | Descrição |
| :--- | :--- |
| [`Makefile`](Makefile) | Automação de compilação, execução e limpeza com flags dedicadas de otimização. |
| [`medir.c`](medir.c) | Exemplo introdutório de medição de ciclos de clock utilizando a instrução de hardware `RDTSC`. |
| [`benchmark_avx.c`](benchmark_avx.c) | Comparativo de soma de array ($N = 2^{20}$) entre processamento **Escalar** e **Vetorizado (AVX-512)**. |
| [`benchmark_unroll.c`](benchmark_unroll.c) | Análise de **Loop Unrolling** e múltiplos acumuladores para explorar paralelismo no nível de instrução (ILP). |
| [`benchmark_fma.c`](benchmark_fma.c) | Multiplicação de matrizes $1024 \times 1024$ comparando algoritmo ingênuo ($i\text{-}j\text{-}k$), **AVX-512 + FMA + Loop Interchange** ($i\text{-}k\text{-}j$) e **AVX-512 + FMA + OpenMP**. |

---

## 🔬 Detalhamento Técnico dos Módulos

### 1. `medir.c` — Medição de Ciclos de Clock com RDTSC
* **Objetivo**: Demonstrar a captura da latência real de execução em ciclos de clock da CPU sem sobrecarga (*overhead*) de chamadas do sistema operacional.
* **Técnicas & Intrínsecos**:
  * `__rdtsc()`: Lê o contador de carimbo de tempo (*Time Stamp Counter* - TSC) de 64 bits do processador via `<x86intrin.h>`.
  * `volatile int i`: Garante que a variável do laço não seja eliminada pelo otimizador (*Dead Code Elimination*).

### 2. `benchmark_avx.c` — Vetorização com AVX-512
* **Objetivo**: Comparar a redução de soma em um vetor de $1.048.576$ elementos ($2^{20}$ floats, $\approx 4\text{ MB}$).
* **Técnicas & Intrínsecos**:
  * `aligned_alloc(64, ...)`: Alocação de memória alinhada em fronteiras de 64 bytes (tamanho de uma linha de cache L1 e de um registrador ZMM de 512 bits).
  * `rdtsc_fence()` com `_mm_lfence()`: Barreira de serialização de pipeline que impede que instruções fora de ordem (*Out-of-Order Execution*) ultrapassem o ponto de medição do `RDTSC`.
  * `_mm512_load_ps()`: Carrega 16 valores `float` (32 bits cada) em um único registrador vetorial ZMM (512 bits).
  * `_mm512_add_ps()`: Executa 16 adições em ponto flutuante em paralelo por ciclo de instrução.
  * `_mm512_reduce_add_ps()`: Redução horizontal otimizada por hardware, somando as 16 vias (*lanes*) do registrador vetorial.

### 3. `benchmark_unroll.c` — Loop Unrolling & Múltiplos Acumuladores (ILP)
* **Objetivo**: Demonstrar como quebrar a cadeia de dependência de dados (*Data Dependency Chain*) para saturar as portas de execução da CPU.
* **Estratégias comparadas**:
  1. **Escalar**: 1 float processado por iteração.
  2. **AVX-512 (1 Acumulador)**: Processa 16 floats por iteração, mas sofre com a latência de throughput da instrução de adição vetorial (cada adição aguarda o resultado anterior no mesmo registrador `acc`).
  3. **AVX-512 (4 Acumuladores)**: Desenrola o laço em $4 \times 16 = 64$ floats por iteração utilizando 4 registradores independentes (`acc0`, `acc1`, `acc2`, `acc3`). Isso permite que a CPU execute múltiplas instruções de carga e soma concorrentemente via *Instruction-Level Parallelism* (ILP), finalizando com uma redução em árvore binária.

### 4. `benchmark_fma.c` — Multiplicação Matricial, FMA & OpenMP
* **Objetivo**: Multiplicação de matrizes densas $1024 \times 1024$ ($2 \times 1024^3 = 2.147.483.648$ FLOPs $\approx 2.15\text{ GFLOPs}$).
* **Estratégias comparadas**:
  1. **Escalar Ingênuo ($i\text{-}j\text{-}k$)**: Acesso à matriz $B$ com salto de linha (*stride* $N$), gerando elevada taxa de *cache misses* e baixo throughput de memória.
  2. **AVX-512 + FMA + Loop Interchange ($i\text{-}k\text{-}j$)**:
     * **Loop Interchange**: Reordena os laços para percorrer as matrizes $B$ e $C$ de forma contígua em memória (*stride* 1), explorando a localidade espacial e os *hardware prefetchers*.
     * **Broadcast (`_mm512_set1_ps`)**: Replica o escalar $A[i][k]$ em todas as 16 vias do registrador ZMM.
     * **Fused Multiply-Add (`_mm512_fmadd_ps`)**: Calcula $(A_{ik} \times B_{kj}) + C_{ij}$ em uma única instrução com um único arredondamento (32 FLOPs por instrução).
  3. **AVX-512 + FMA + OpenMP**: Distribui o laço externo $i$ entre todos os núcleos/threads disponíveis via `#pragma omp parallel for schedule(static)`.

---

## ⚙️ Requisitos de Compilação e Hardware

* **Arquitetura**: Processador x86-64 com suporte a **AVX-512F**, **AVX-512DQ** e **FMA** (ex.: Intel Skylake-X, Ice Lake, Tiger Lake, Rocket Lake, Sapphire Rapids ou AMD Zen 4/5).
* **Compilador**: GCC 9+ ou Clang 10+ com suporte a OpenMP (`libgomp`).
* **Flags de Otimização Utilizadas**:
  * `-O3`: Nível máximo de otimização padrão.
  * `-march=native`: Habilita o conjunto de instruções nativo da CPU hospedeira.
  * `-fno-tree-vectorize`: Desabilita a autovetorização do compilador para permitir uma comparação justa com o baseline escalar manual.
  * `-mavx512f -mavx512dq`: Habilita instruções AVX-512 Foundation e Doubleword/Quadword.
  * `-mfma`: Habilita suporte a instruções Fused Multiply-Add.
  * `-fopenmp`: Habilita suporte a multithreading OpenMP.

---

## 🛠️ Compilação e Execução

### Utilizando o Makefile (Recomendado)

```bash
# Compilar todos os binários
make

# Executar toda a suite de testes sequencialmente
make run

# Limpar os binários gerados
make clean
```

### Compilação Manual (Passo a Passo)

```bash
# 1. Medição simples
gcc -O3 -march=native -Wall -fno-tree-vectorize medir.c -o medir

# 2. Benchmark AVX-512
gcc -O3 -march=native -Wall -fno-tree-vectorize -mavx512f -mavx512dq benchmark_avx.c -o benchmark_avx

# 3. Benchmark Unrolling (ILP)
gcc -O3 -march=native -Wall -fno-tree-vectorize -mavx512f -mavx512dq benchmark_unroll.c -o benchmark_unroll

# 4. Benchmark FMA Matrizes (com OpenMP)
gcc -O3 -march=native -Wall -fno-tree-vectorize -mavx512f -mavx512dq -mfma -fopenmp benchmark_fma.c -o benchmark_fma
```

---

## 📊 Resultados e Análise de Desempenho

> **Ambiente de Teste**: Intel(R) Core(TM) i7-1165G7 @ 2.80GHz (4 Núcleos / 8 Threads, AVX-512 habilitado).

### 1. Soma de Vetor (`benchmark_avx`)
*Elemento processados: 1.048.576 floats ($4\text{ MB}$)*

```text
Elementos processados: 1048576
Escalar  | Resultado: 1048576 | Ciclos:    7632108 | Ciclos/elem: 7.28
AVX-512  | Resultado: 1048576 | Ciclos:    1514318 | Ciclos/elem: 1.44
Speedup  | 5.04x mais rápido
```

---

### 2. Desenrolamento de Laço & ILP (`benchmark_unroll`)
*Elemento processados: 1.048.576 floats*

```text
Estratégia           | Resultado  | Ciclos Totais | Ciclos/elem | Speedup
---------------------|------------|---------------|-------------|--------
1. Escalar           |    1048576 |       7000469 |        6.68 | 1.00x
2. AVX-512 (1x Acc)  |    1048576 |       1108011 |        1.06 | 6.32x
3. AVX-512 (4x Acc)  |    1048576 |        678515 |        0.65 | 10.32x
```
* **Insight**: A utilização de 4 acumuladores independentes reduz o tempo por elemento para **0.65 ciclos**, superando o throughput de 1 acumulador devido à eliminação do gargalo de dependência na pipeline de execução vetorial.

---

### 3. Multiplicação de Matriz Dense ($1024 \times 1024$) (`benchmark_fma`)
*Total de Operações: $2.147.483.648\text{ FLOPs}$ ($2.15\text{ GFLOPs}$)*

```text
Matrizes: 1024x1024 | Total FLOPs: 2147483648 | Threads OpenMP: 8

Estratégia               | Ciclos Totais | FLOPs/Ciclo | Speedup
-------------------------|---------------|-------------|--------
1. Escalar (1 thread)    |    6194722896 |        0.35 | 1.00x
2. AVX-512 FMA (1 thread)|     227861463 |        9.42 | 27.19x
3. AVX-512 FMA + OpenMP  |      77747574 |       27.62 | 79.68x
```
* **Insight**:
  * **Loop Interchange + AVX-512 + FMA**: Eleva o throughput de **0.35 FLOPs/ciclo** para **9.42 FLOPs/ciclo** (ganho de **27.19x**), combinando acesso sequencial à cache com a densidade computacional da instrução FMA (32 FLOPs por instrução de 512 bits).
  * **OpenMP Multi-threading**: Ao escalar entre os 8 threads lógicos da CPU, o tempo cai para **77 milhões de ciclos** (**27.62 FLOPs/ciclo**), atingindo um speedup total de **~80x** em relação ao algoritmo escalar inicial.

---

## 📌 Resumo das Técnicas Aplicadas

| Técnica | Mecanismo | Impacto Principal |
| :--- | :--- | :--- |
| **Alinhamento de Memória (64B)** | `aligned_alloc(64, ...)` | Evita penalidades de *unaligned memory access* e garante alinhamento com a linha de cache L1. |
| **Serialização de Pipeline** | `_mm_lfence()` + `__rdtsc()` | Medições de tempo confiáveis sem interferência da execução fora de ordem (*OoO*). |
| **Vetorização SIMD (AVX-512)** | Registradores ZMM (512 bits) | Processa 16 valores `float` simultaneamente por instrução. |
| **Loop Unrolling & ILP** | Múltiplos acumuladores (`acc0`-`acc3`) | Oculta a latência das unidades funcionais e satura as portas de execução. |
| **Loop Interchange ($i\text{-}k\text{-}j$)** | Reordenação de loops | Acesso contíguo à memória (stride 1) eliminando *cache misses*. |
| **Fused Multiply-Add (FMA)** | `_mm512_fmadd_ps` | Duplica o throughput aritmético ($(A \times B) + C$) com apenas 1 ciclo de arredondamento. |
| **Paralelismo Multithread** | `#pragma omp parallel for` | Divide uniformemente a carga computacional entre os núcleos da CPU. |

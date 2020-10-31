//===-- bench_dual - test dual class ----------------------------*- C++ -*-===//
//
// Part of the cppduals Project
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c)2019 Michael Tesch. tesch1@gmail.com
//
#include <iostream>
#include "benchmark/benchmark.h"
#include "type_name.hpp"
#include <Eigen/Core> // for SimdInstructionSetsInUse()
#include <duals/dual>

#define N_RANGE ->RangeMultiplier(8)->Range(2, 1<<13) // 2 - 2048


template <class Rt> void B_Add(benchmark::State& state) {
  int N = state.range(0);
  std::vector<Rt> a(N);
  std::vector<Rt> b(N);
  std::vector<Rt> c(N);
  for (auto _ : state) {
    for (int i = 0; i < N; i++)
      a[i] = b[i] + c[i];
    benchmark::ClobberMemory(); // Force a to be written to memory.
  }
}

template <class Rt> void B_Sub(benchmark::State& state) {
  int N = state.range(0);
  std::vector<Rt> a(N);
  std::vector<Rt> b(N);
  std::vector<Rt> c(N);
  for (auto _ : state) {
    for (int i = 0; i < N; i++)
      a[i] = b[i] - c[i];
    benchmark::ClobberMemory(); // Force a to be written to memory.
  }
}

template <class Rt> void B_Mul(benchmark::State& state) {
  int N = state.range(0);
  std::vector<Rt> a(N);
  std::vector<Rt> b(N);
  std::vector<Rt> c(N);
  for (auto _ : state) {
    for (int i = 0; i < N; i++)
      a[i] = b[i] * c[i];
    benchmark::ClobberMemory(); // Force a to be written to memory.
  }
}

template <class Rt> void B_Div(benchmark::State& state) {
  int N = state.range(0);
  std::vector<Rt> a(N);
  std::vector<Rt> b(N);
  std::vector<Rt> c(N);
  for (auto _ : state) {
    for (int i = 0; i < N; i++)
      a[i] = b[i] / c[i];
    benchmark::ClobberMemory(); // Force a to be written to memory.
  }
}

BENCHMARK_TEMPLATE(B_Add, duals::dualf) N_RANGE;
BENCHMARK_TEMPLATE(B_Add, std::complex<float>) N_RANGE;
//BENCHMARK_TEMPLATE(B_Add, std::complex<duals::dualf>) N_RANGE;

BENCHMARK_TEMPLATE(B_Sub, duals::dualf) N_RANGE;
BENCHMARK_TEMPLATE(B_Sub, std::complex<float>) N_RANGE;

BENCHMARK_TEMPLATE(B_Mul, duals::dualf) N_RANGE;
BENCHMARK_TEMPLATE(B_Mul, std::complex<float>) N_RANGE;

BENCHMARK_TEMPLATE(B_Div, duals::dualf) N_RANGE;
BENCHMARK_TEMPLATE(B_Div, std::complex<float>) N_RANGE;

#define QUOTE(...) STRFY(__VA_ARGS__)
#define STRFY(...) #__VA_ARGS__

int main(int argc, char** argv)
{
  std::cout << "OPT_FLAGS=" << QUOTE(OPT_FLAGS) << "\n";
  std::cout << "INSTRUCTIONSET=" << Eigen::SimdInstructionSetsInUse() << "\n";
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
}

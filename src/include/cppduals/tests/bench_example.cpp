//===-- bench_example - benchmark the examples ---------------*- C++ -*-===//
//
// Part of the cppduals project.
// https://gitlab.com/tesch1/cppduals
//
// See https://gitlab.com/tesch1/cppduals/blob/master/LICENSE.txt for
// license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c)2019 Michael Tesch. tesch1@gmail.com
//

#include <complex>
#include <duals/dual>
#include "benchmark/benchmark.h"

using namespace duals;

template <class T> T   f(T x) { return pow(x,pow(x,x)); }
template <class T> T  df(T x) { return pow(x,-1. + x + pow(x,x)) * (1. + x*log(x) + x*pow(log(x),2.)); }
template <class T> T ddf(T x) { return (pow(x,pow(x,x)) * pow(pow(x,x - 1.) + pow(x,x)*log(x)*(log(x) + 1.), 2.) +
                                        pow(x,pow(x,x)) * (pow(x,x - 1.) * log(x) +
                                                           pow(x,x - 1.) * (log(x) + 1.) +
                                                           pow(x,x - 1.) * ((x - 1.)/x + log(x)) +
                                                           pow(x,x) * log(x) * pow(log(x) + 1., 2.) )); }

template <class T>
void F(benchmark::State& state) {
  T x = state.range(0);
  for (auto _ : state) { benchmark::DoNotOptimize(f(x)); }
}

template <class T>
void DF(benchmark::State& state) {
  T x = state.range(0);
  for (auto _ : state) { benchmark::DoNotOptimize(df(x)); }
}

template <class T>
void DDF(benchmark::State& state) {
  T x = state.range(0);
  for (auto _ : state) { benchmark::DoNotOptimize(ddf(x)); }
}

template <class T>
void dF(benchmark::State& state) {
  T x = state.range(0);
  for (auto _ : state) { benchmark::DoNotOptimize(f(dual<T>(x,1))); }
}

template <class T>
void ddF(benchmark::State& state) {
  T x = state.range(0);
  for (auto _ : state) {
    benchmark::DoNotOptimize(f(duals::dual<duals::dual<T>>(x+1_e,1+0_e)));
  }
}

BENCHMARK_TEMPLATE(F, float) ->Arg(2);  //  V_RANGE(1,NF)
BENCHMARK_TEMPLATE(DF, float)->Arg(2);  //  V_RANGE(1,NF)
BENCHMARK_TEMPLATE(DDF, float)->Arg(2); //  V_RANGE(1,NF)
BENCHMARK_TEMPLATE(dF, float)->Arg(2);  //  V_RANGE(1,NF)
BENCHMARK_TEMPLATE(ddF, float)->Arg(2); //  V_RANGE(1,NF)

#define QUOTE(...) STRFY(__VA_ARGS__)
#define STRFY(...) #__VA_ARGS__

int main(int argc, char** argv)
{
  std::cout << "OPT_FLAGS=" << QUOTE(OPT_FLAGS) << "\n";
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
}

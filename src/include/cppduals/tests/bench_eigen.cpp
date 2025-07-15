//===-- bench_dual - test dual class ----------------------------*- C++ -*-===//
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

#include <duals/dual_eigen>
#include <iostream>
#include <fstream>
#include <complex>
#include <memory>

#include "type_name.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>

#include "eexpokit/padm.hpp"
#include "eexpokit/chbv.hpp"
#include "eexpokit/expv.hpp"
//#include "eexpokit/mexpv.hpp"
#include "benchmark/benchmark.h"

using namespace duals;

template< class T > struct type_identity { typedef T type; };

/* encode the type into an integer for benchmark output */
template<typename Tp> struct type_num { /* should fail */ };
template<> struct type_num<float>     { static constexpr int id = 1; };
template<> struct type_num<double>     { static constexpr int id = 2; };
template<> struct type_num<long double> { static constexpr int id = 3; };
template<typename Tp> struct type_num<std::complex<Tp>> { static constexpr int id = 10 + type_num<Tp>::id; };
template<typename Tp> struct type_num<duals::dual<Tp>> { static constexpr int id = 100 + type_num<Tp>::id; };

using duals::dualf;
using duals::duald;
typedef std::complex<double> complexd;
typedef std::complex<float> complexf;
typedef std::complex<duald> cduald;
typedef std::complex<dualf> cdualf;
template <class T> using MatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

#if 0
#define V_RANGE(V,NF) ->Arg(V*4/NF)->Arg(V*32/NF)->Arg(V*256/NF)->Arg(V*2048/NF)->Arg(V*1)->Complexity()
#else
#define V_RANGE(V,NF) ->Arg(V*1024/NF)
#endif

template <class Rt, class U> void B_VecVecAddCXX(benchmark::State& state) {
  int N = state.range(0);
  std::vector<Rt> a(N);
  std::vector<Rt> b(N);
  std::vector<Rt> c(N);
  for (auto _ : state) {
    for (int i = 0; i < N; i++)
      a[i] = b[i] + b[i];
    benchmark::ClobberMemory(); // Force a to be written to memory.
  }
  state.counters["type"] = type_num<Rt>::id;
  state.SetComplexityN(state.range(0));
}

// measure Eigen's vec-vec multiplication
template <class Rt, class U> void B_VecVecAdd(benchmark::State& state) {
  int N = state.range(0);
  MatrixX<Rt> A = MatrixX<Rt>::Random(N, 1);
  MatrixX<Rt> B = MatrixX<Rt>::Random(N, 1);
  MatrixX<Rt> C = MatrixX<Rt>::Random(N, 1);
  for (auto _ : state) {
    B = A + A;
    benchmark::ClobberMemory(); // Force a to be written to memory.
  }
  state.counters["type"] = type_num<Rt>::id;
  state.SetComplexityN(state.range(0));
}

// measure Eigen's vec-vec multiplication
template <class Rt, class U> void B_VecVecSub(benchmark::State& state) {
  int N = state.range(0);
  MatrixX<Rt> A = MatrixX<Rt>::Random(N, 1);
  MatrixX<Rt> B = MatrixX<Rt>::Random(N, 1);
  MatrixX<Rt> C = MatrixX<Rt>::Random(N, 1);
  for (auto _ : state) {
    B = A - A;
    benchmark::ClobberMemory(); // Force a to be written to memory.
  }
  state.counters["type"] = type_num<Rt>::id;
  state.SetComplexityN(state.range(0));
}

template <class Rt, class U> void B_VecVecMulCXX(benchmark::State& state) {
  int N = state.range(0);
  std::vector<Rt> a(N);
  std::vector<Rt> b(N);
  std::vector<Rt> c(N);
  for (auto _ : state) {
    for (int i = 0; i < N; i++)
      b[i] = a[i] * a[i];
    benchmark::ClobberMemory(); // Force a to be written to memory.
  }
  state.counters["type"] = type_num<Rt>::id;
  state.SetComplexityN(state.range(0));
}

// measure Eigen's vec-vec multiplication
template <class Rt, class U> void B_VecVecMul(benchmark::State& state) {
  int N = state.range(0);
  MatrixX<Rt> A = MatrixX<Rt>::Random(N, 1);
  MatrixX<Rt> B = MatrixX<Rt>::Random(N, 1);
  MatrixX<Rt> C = MatrixX<Rt>::Random(N, 1);
  for (auto _ : state) {
    B = A.array() * C.array();
    benchmark::ClobberMemory(); // Force a to be written to memory.
  }
  state.counters["type"] = type_num<Rt>::id;
  state.SetComplexityN(state.range(0));
}

// measure Eigen's vec-vec multiplication
template <class Rt, class U> void B_VecVecDiv(benchmark::State& state) {
  int N = state.range(0);
  MatrixX<Rt> A = MatrixX<Rt>::Random(N, 1);
  MatrixX<Rt> B = MatrixX<Rt>::Random(N, 1);
  MatrixX<Rt> C = MatrixX<Rt>::Random(N, 1);
  for (auto _ : state) {
    C = A.array() / B.array();
    benchmark::ClobberMemory(); // Force a to be written to memory.
  }
  state.counters["type"] = type_num<Rt>::id;
  state.SetComplexityN(state.range(0));
}

// measure Eigen's matrix-matrix multiplication
template <class T, class U> void B_MatMat(benchmark::State& state) {
  int N = state.range(0);
  typedef typename Eigen::ScalarBinaryOpTraits<T, U>::ReturnType R;
  MatrixX<T> A = MatrixX<T>::Random(N, N);
  MatrixX<U> B = MatrixX<U>::Random(N, N);
  MatrixX<R> C = MatrixX<R>::Random(N, N);
  for (auto _ : state) {
    C.noalias() = A * B;
    benchmark::ClobberMemory(); // Force c to be written to memory.
  }

  state.counters["type"] = type_num<T>::id;
  state.SetComplexityN(state.range(0));
}

// measure compiler's matrix-matrix multiplication
template <class Rt, class U> void B_MatMatCXX(benchmark::State& state) {
  int N = state.range(0);
  std::vector<Rt> a(N*N);
  std::vector<Rt> b(N*N);
  std::vector<Rt> c(N*N);

  for (auto _ : state) {
    state.PauseTiming();
    a.assign(N*N,1.1);
    b.assign(N*N,2.2);
    c.assign(N*N,0.);
    state.ResumeTiming();

    for(int i=0; i<N; ++i)
      for(int j=0; j<N; ++j)
        for(int k=0; k<N; ++k)
          c[i*N+j] += a[i*N+k] * b[k*N+j];

    benchmark::ClobberMemory(); // Force a to be written to memory.
  }

  state.counters["type"] = type_num<Rt>::id;
  state.SetComplexityN(state.range(0));
}

// measure Eigen's matrix-matrix solve
template <class Rt, class U> void B_MatDiv(benchmark::State& state) {
  int N = state.range(0);
  MatrixX<Rt> A = MatrixX<Rt>::Random(N, N);
  MatrixX<Rt> B = MatrixX<Rt>::Random(N, N);
  MatrixX<Rt> C = MatrixX<Rt>::Zero(N, N);

  for (auto _ : state) {
    C = B.partialPivLu().solve(A); // C = A / B
    benchmark::ClobberMemory();
  }

  state.counters["type"] = type_num<Rt>::id;
  state.SetComplexityN(state.range(0));
}

// measure eigen's matrix-vector multiplication
template <class Rt, class U> void B_MatVec(benchmark::State& state) {
  int N = state.range(0);
  MatrixX<Rt> A = MatrixX<Rt>::Random(N, N);
  MatrixX<Rt> b = MatrixX<Rt>::Random(N, 1);
  MatrixX<Rt> c = MatrixX<Rt>::Random(N, 1);
  for (auto _ : state) {
    c = A * b;
    benchmark::ClobberMemory();
  }

  state.counters["type"] = type_num<Rt>::id;
  state.SetComplexityN(state.range(0));
}

#define MAKE_BM_SIMPLE(TYPE1,TYPE2,NF)                                  \
  BENCHMARK_TEMPLATE(B_VecVecAdd, TYPE1,TYPE2) V_RANGE(4,NF);           \
  BENCHMARK_TEMPLATE(B_VecVecSub, TYPE1,TYPE2) V_RANGE(4,NF);           \
  BENCHMARK_TEMPLATE(B_VecVecMul, TYPE1,TYPE2) V_RANGE(4,NF);           \
  BENCHMARK_TEMPLATE(B_VecVecDiv, TYPE1,TYPE2) V_RANGE(4,NF);           \
  BENCHMARK_TEMPLATE(B_MatVec, TYPE1,TYPE2) V_RANGE(4,NF);              \
  BENCHMARK_TEMPLATE(B_MatMat, TYPE1,TYPE2) V_RANGE(1,NF);              \
  BENCHMARK_TEMPLATE(B_MatDiv, TYPE1,TYPE2) V_RANGE(1,NF)

#define MAKE_BENCHMARKS(TYPE1,TYPE2,NF)                                 \
  MAKE_BM_SIMPLE(TYPE1,TYPE2,NF)

//  BENCHMARK_TEMPLATE(B_VecVecMulCXX, TYPE1,TYPE2) V_RANGE(4,NF);
//  BENCHMARK_TEMPLATE(B_MatMatCXX, TYPE1,TYPE2) V_RANGE(1,NF);

#if 1
MAKE_BENCHMARKS(float, float, 1);
MAKE_BENCHMARKS(complexf, complexf,2);
MAKE_BM_SIMPLE(dualf, dualf,2);
MAKE_BM_SIMPLE(cdualf, cdualf,4);
#else
MAKE_BENCHMARKS(double, double,1);
MAKE_BENCHMARKS(complexd, complexd,2);
MAKE_BM_SIMPLE(duald, duald,2);
MAKE_BM_SIMPLE(cduald, cduald,4);
#endif

#define QUOTE(...) STRFY(__VA_ARGS__)
#define STRFY(...) #__VA_ARGS__

int main(int argc, char** argv)
{
#ifndef EIGEN_VECTORIZE
  //static_assert(false, "no vectorization?");
#endif
  std::cout << "OPT_FLAGS=" << QUOTE(OPT_FLAGS) << "\n";
  std::cout << "INSTRUCTIONSET=" << Eigen::SimdInstructionSetsInUse() << "\n";
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
}

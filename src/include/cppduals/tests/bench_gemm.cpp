//===-- bench_gemm - benchmark dual m*m -----------------------*- C++ -*-===//
//
// Part of the cppduals Project
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c)2019 Michael Tesch. tesch1@gmail.com
//
#if defined(__APPLE__) && defined(__clang__)
#include <Accelerate/Accelerate.h>

#else

#ifdef EIGEN_LAPACKE
#include <Eigen/src/misc/lapacke.h>
#else
#include <lapacke.h>
#endif

extern "C" {
  //#include <cblas.h>
  //#include <cblas_openblas.h>
  #include CBLAS_HEADER
}
#endif // defined(__APPLE__) && defined(__clang__)

#include <iostream>
#include <fstream>
#include <complex>
#include <memory>

#include "type_name.hpp"
#include <duals/dual_eigen>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "benchmark/benchmark.h"

using namespace duals;

template< class T > struct type_identity { typedef T type; };

#include <unsupported/Eigen/MatrixFunctions>


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
#define V_RANGE(V,NF) ->Arg(V*64/NF)->Arg(V*128/NF)->Arg(V*256/NF)->Arg(V*512/NF)->Arg(V*1024/NF) ->Arg(V*2048/NF)
#endif

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

  state.SetComplexityN(state.range(0));
}

template <class T, typename std::enable_if<!duals::is_dual<T>::value>::type* = nullptr>
void matrix_multiplcation(T *A, int Awidth, int Aheight,
                          T *B, int Bwidth, int Bheight,
                          T *AB, bool tA, bool tB,
                          typename type_identity<T>::type beta)
{
  int A_height = tA ? Awidth  : Aheight;
  int A_width  = tA ? Aheight : Awidth;
#ifndef NDEBUG
  int B_height = tB ? Bwidth  : Bheight;
#endif
  int B_width  = tB ? Bheight : Bwidth;
  int m = A_height;
  int n = B_width;
  int k = A_width;
  // Error, width and height should match!
  assert(A_width == B_height);
  int lda = tA ? m : k;
  int ldb = tB ? k : n;
#define TRANSPOSE(X) ((X) ? CblasTrans : CblasNoTrans)
  // http://www.netlib.org/lapack/explore-html/d7/d2b/dgemm_8f.html
  if (!is_complex<T>::value) {
    if (sizeof(T) == sizeof(float))
      cblas_sgemm(CblasColMajor, TRANSPOSE(tA), TRANSPOSE(tB),
                  m, n, k, 1.0, (float *)A, lda, (float *)B, ldb,
                  std::real(beta), (float *)AB, n);
    else
      cblas_dgemm(CblasColMajor, TRANSPOSE(tA), TRANSPOSE(tB),
                  m, n, k, 1.0, (double *)A, lda, (double *)B, ldb,
                  std::real(beta), (double *)AB, n);
  }
  else {
    std::complex<float> alphaf(1,0);
    std::complex<double> alpha(1,0);
    if (Eigen::NumTraits<T>::digits10() < 10)
      cblas_cgemm(CblasColMajor, TRANSPOSE(tA), TRANSPOSE(tB),
                  m, n, k, &alphaf, A, lda, B, ldb, &beta, AB, n);
    else
      cblas_zgemm(CblasColMajor, TRANSPOSE(tA), TRANSPOSE(tB),
                  m, n, k, &alpha, A, lda, B, ldb, &beta, AB, n);
  }
#undef TRANSPOSE
}

template <class T, typename std::enable_if<duals::is_dual<T>::value>::type* = nullptr>
void matrix_multiplcation(T *A, int Awidth, int Aheight,
                          T *B, int Bwidth, int Bheight,
                          T *AB, bool tA, bool tB,
                          typename type_identity<T>::type beta)
{ /* nothing */
}

// measure BLAS matrix-matrix multiplication
template <class Rt> void B_MatMatBLAS(benchmark::State& state) {
  int N = state.range(0);
  MatrixX<Rt> A = MatrixX<Rt>::Random(N, N);
  MatrixX<Rt> B = MatrixX<Rt>::Random(N, N);
  MatrixX<Rt> C = MatrixX<Rt>::Random(N, N);
  MatrixX<Rt> D = A*B;
  for (auto _ : state) {
    matrix_multiplcation(A.data(), A.cols(), A.rows(),
                         B.data(), B.cols(), B.rows(),
                         C.data(), false, false, (Rt)0.);
    benchmark::ClobberMemory(); // Force a to be written to memory.
  }

  double err = (double)rpart((D - C).norm() / D.norm());
  if (err > 1e-6)
    state.SkipWithError("BLAS matmat error");

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
  BENCHMARK_TEMPLATE(B_MatMat, TYPE1,TYPE2) V_RANGE(1,NF)

#define MAKE_BENCHMARKS(TYPE1,TYPE2,NF)                                 \
  MAKE_BM_SIMPLE(TYPE1,TYPE2,NF);                                       \
  BENCHMARK_TEMPLATE(B_MatMatBLAS, TYPE1) V_RANGE(1,NF)

//  BENCHMARK_TEMPLATE(B_MatMatBLAS, TYPE1) V_RANGE(1,2*NF);
//  BENCHMARK_TEMPLATE(B_VecVecMulCXX, TYPE1,TYPE2) V_RANGE(4,NF);
//  BENCHMARK_TEMPLATE(B_MatMatCXX, TYPE1,TYPE2) V_RANGE(1,NF);

MAKE_BENCHMARKS(float, float, 1);
MAKE_BENCHMARKS(complexf, complexf,2);
//MAKE_BM_SIMPLE(dualf, float,2); TODO
MAKE_BM_SIMPLE(dualf, dualf,2);
//MAKE_BM_SIMPLE(cdualf, cdualf,2);
MAKE_BM_SIMPLE(cdualf, cdualf,4);

#if HAVE_BOOST
#include <audi/audi.hpp>
MAKE_BM_SIMPLE(audi::gdual<float>,2);
#endif

// novelty:
//MAKE_BM_SIMPLE(float, complexf,2);
//MAKE_BM_SIMPLE(complexf, float,2);

MAKE_BENCHMARKS(double, double, 1);
MAKE_BENCHMARKS(complexd, complexd,2);
MAKE_BM_SIMPLE(duald, duald,2);
MAKE_BM_SIMPLE(cduald, cduald,4);

#define QUOTE(...) STRFY(__VA_ARGS__)
#define STRFY(...) #__VA_ARGS__

int main(int argc, char** argv)
{
#ifndef EIGEN_VECTORIZE
  static_assert(false, "no vectorization?");
#endif
#ifndef NDEBUG
  static_assert(false, "NDEBUG to benchmark?");
#endif
  std::cout << "OPT_FLAGS=" << QUOTE(OPT_FLAGS) << "\n";
  std::cout << "INSTRUCTIONSET=" << Eigen::SimdInstructionSetsInUse() << "\n";
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
}

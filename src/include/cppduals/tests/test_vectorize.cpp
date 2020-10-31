//===-- test_vectorize.cpp - test duals/dual_eigen -------------*- C++ -*-===//
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
/**
 * \file test_vectorize Dual number Eigen vectorization op tests
 *
 * (c)2019 Michael Tesch. tesch1@gmail.com
 */

#include "type_name.hpp"
#include <duals/dual_eigen>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/StdVector>
#include "gtest/gtest.h"

using duals::rpart;
using duals::dpart;
using duals::dualf;
using duals::duald;
using duals::dualld;
using duals::hyperdualf;
using duals::hyperduald;
using duals::hyperdualld;
using duals::is_dual;
using duals::is_complex;
using duals::dual_traits;
using namespace duals::literals;

typedef std::complex<double> complexd;
typedef std::complex<float> complexf;
typedef std::complex<duald> cduald;
typedef std::complex<dualf> cdualf;

template <class eT, int N=Eigen::Dynamic, int K = N> using emtx = Eigen::Matrix<eT, N, K>;
template <class eT> using smtx = Eigen::SparseMatrix<eT>;

template <int N=2, int K = N> using ecf = Eigen::Matrix<complexf, N, K> ;
template <int N=2, int K = N> using edf = Eigen::Matrix<dualf, N, K> ;
template <int N=2, int K = N> using ecdf = Eigen::Matrix<cdualf, N, K> ;

#define _EXPECT_TRUE(...)  {typedef __VA_ARGS__ tru; EXPECT_TRUE(tru::value); static_assert(tru::value, "sa"); }
#define _EXPECT_FALSE(...) {typedef __VA_ARGS__ fal; EXPECT_FALSE(fal::value); static_assert(!fal::value, "sa"); }
#define ASSERT_DEQ(A,B) ASSERT_EQ(rpart(A), rpart(B)); ASSERT_EQ(dpart(A), dpart(B))
#define ASSERT_DNEAR(A,B,tol)                                   \
  ASSERT_NEAR(abs(rpart((A) - (B))),0,abs(rpart(A))*(tol));     \
  ASSERT_NEAR(abs(dpart((A) - (B))),0,abs(dpart(A))*(tol))
#define EXPECT_DEQ(A,B) EXPECT_EQ(rpart(A), rpart(B)); EXPECT_EQ(dpart(A), dpart(B))
#define EXPECT_DNE(A,B) EXPECT_NE(rpart(A), rpart(B)); EXPECT_NE(dpart(A), dpart(B))
#define EXPECT_DNEAR(A,B,tol)                                    \
  EXPECT_NEAR(abs(rpart((A) - (B))),0,abs(rpart(A))*(tol));      \
  EXPECT_NEAR(abs(dpart((A) - (B))),0,abs(dpart(A))*(tol))


template <typename Rt>
void elemwise(int N) {
  std::vector<Rt> a(N);
  std::vector<Rt> b(N);
  std::vector<Rt> cp(N);
  std::vector<Rt> cm(N);
  std::vector<Rt> ct(N);
  std::vector<Rt> cd(N);
  std::vector<Rt> cc(N);
  std::vector<Rt> cca(N);
  std::vector<Rt> ccb(N);
  emtx<Rt> A(N,1);
  emtx<Rt> B(N,1);
  emtx<Rt> Cp(N,1);
  emtx<Rt> Cm(N,1);
  emtx<Rt> Ct(N,1);
  emtx<Rt> Cd(N,1);
  emtx<Rt> Cc(N,1);
  emtx<Rt> Cca(N,1);
  emtx<Rt> Ccb(N,1);
  double tol = rpart(2000*Eigen::NumTraits<Rt>::epsilon());
  using duals::randos::random2;

  Rt sum(0);
  for (int i = 0; i < N; i++) {
    A(i) = a[i] = random2<Rt>();
    B(i) = b[i] = random2<Rt>();
    cp[i] = a[i] + b[i];
    cm[i] = a[i] - b[i];
    ct[i] = a[i] * b[i];
    cd[i] = a[i] / b[i];
    cc[i] = conj(a[i]) + conj(b[i]);
    cca[i] = conj(a[i]) + b[i];
    ccb[i] = a[i] + conj(b[i]);
    sum += a[i];
  }

  Cp = A.array() + B.array();
  Cm = A.array() - B.array();
  Ct = A.array() * B.array();
  Cd = A.array() / B.array();
  Cc = conj(A.array()) + conj(B.array());
  Cca = conj(A.array()) + B.array();
  Ccb = A.array() + conj(B.array());

  for (int i = 0; i < N; i++) {
    ASSERT_DEQ(cp[i], Cp(i)) << "p mismatch at " << i << "\n";
    ASSERT_DEQ(cm[i], Cm(i)) << "m mismatch at " << i << "\n";
    ASSERT_DNEAR(ct[i], Ct(i),tol) << "t mismatch at " << i << "\n";
    ASSERT_DNEAR(cd[i], Cd(i),3*tol) /* why is this so much worse? */
      << "d mismatch at " << i << " " << cd[i] << "|\n"
      << Cd(i) << " != " << a[i] << "/" << b[i] << "\n";
    ASSERT_DEQ(cc[i], Cc(i)) << "c mismatch at " << i << "\n";
    ASSERT_DEQ(cca[i], Cca(i)) << "ca mismatch at " << i << "\n";
    ASSERT_DEQ(ccb[i], Ccb(i)) << "cb mismatch at " << i << "\n";
  }
  ASSERT_DNEAR(sum, A.sum(), N*tol);
  ASSERT_DNEAR(sum, A.sum(), N*tol);
}

TEST(Vector, full_even_dualf) { elemwise<dualf>(512); }
TEST(Vector, full_even_duald) { elemwise<duald>(512); }
TEST(Vector, full_even_cdualf) { elemwise<cdualf>(512); }

TEST(Vector, full_odd_dualf) { elemwise<dualf>(2049); }
TEST(Vector, full_odd_duald) { elemwise<duald>(2049); }
TEST(Vector, full_odd_cdualf) { elemwise<cdualf>(2049); }

TEST(Vector, single_elem_dualf) { elemwise<dualf>(1); }
TEST(Vector, single_elem_duald) { elemwise<duald>(1); }
TEST(Vector, single_elem_cdualf) { elemwise<cdualf>(1); }

TEST(Vector, two_elem_dualf) { elemwise<dualf>(2); }
TEST(Vector, two_elem_duald) { elemwise<duald>(2); }
TEST(Vector, two_elem_cdualf) { elemwise<cdualf>(2); }

#define DBOUT(X)
#define MAKE_MULT_TEST(TYPE1, TYPE2, FIX, SIZE)                     \
TEST(MatMult, TYPE1##_##TYPE2##_##SIZE) {                           \
  typedef TYPE1 T1;                                                 \
  typedef TYPE2 T2;                                                 \
  typedef decltype(TYPE1() * TYPE2()) T3;                           \
  using duals::rpart;                                               \
  using duals::conj;                                                \
  /*using std::conj;*/                                              \
  using duals::randos::random2;                                     \
                                                                    \
  double tol = rpart(20*Eigen::NumTraits<T1>::epsilon());           \
  static const Eigen::Index N = FIX;                                \
  static const int n = SIZE;                                        \
  emtx<T1,N> A(n,n);                                                \
  emtx<T2,N> B(n,n);                                                \
  emtx<T3,N> C(n,n);                                                \
  emtx<T3,N> D(n,n);                                                \
                                                                    \
  for (Eigen::Index i = 0; i < A.size(); i++) {                     \
    A.data()[i] = random2<T1>();                                    \
    B.data()[i] = random2<T2>();                                    \
  }                                                                 \
                                                                    \
  C.setZero();                                                      \
  for(int i=0; i<n; ++i)                                            \
    for(int j=0; j<n; ++j)                                          \
      for(int k=0; k<n; ++k)                                        \
        C.data()[j*n+i] += A.data()[k*n+i] * B.data()[j*n+k];       \
                                                                    \
  DBOUT(std::cerr << "--------------------------\n");               \
  D = A*B;                                                          \
  ASSERT_NEAR((double)rpart(D - C).norm(), 0, rpart(C).norm() * tol) << "r|a*b" << C << "\n" << D; \
  ASSERT_NEAR((double)dpart(D - C).norm(), 0, dpart(C).norm() * tol) << "d|a*b" << C << "\n" << D; \
                                                                    \
  C.setZero();                                                      \
  for(int i=0; i<n; ++i)                                            \
    for(int j=0; j<n; ++j)                                          \
      for(int k=0; k<n; ++k)                                        \
        C.data()[j*n+i] += A.data()[k*n+i] * conj(B.data()[k*n+j]); \
                                                                    \
  DBOUT(std::cerr << CRED "  a*b'" CRESET "\n");                    \
  D = A*B.adjoint();                                                \
  ASSERT_NEAR((double)rpart(D - C).norm(), 0, rpart(C).norm() * tol) << "r|a*b'"; \
  ASSERT_NEAR((double)dpart(D - C).norm(), 0, dpart(C).norm() * tol) << "d|a*b'"; \
                                                                    \
  C.setZero();                                                      \
  for(int i=0; i<n; ++i)                                            \
    for(int j=0; j<n; ++j)                                          \
      for(int k=0; k<n; ++k)                                        \
        C.data()[j*n+i] += conj(A.data()[i*n+k]) * B.data()[j*n+k]; \
                                                                    \
  DBOUT(std::cerr << CRED "  a'*b" CRESET "\n");                    \
  D = A.adjoint()*B;                                                \
  ASSERT_NEAR((double)rpart(D - C).norm(), 0, rpart(C).norm() * tol) << "r|a'*b"; \
  ASSERT_NEAR((double)dpart(D - C).norm(), 0, dpart(C).norm() * tol) << "d|a'*b"; \
                                                                    \
  C.setZero();                                                      \
  for(int i=0; i<n; ++i)                                            \
    for(int j=0; j<n; ++j)                                          \
      for(int k=0; k<n; ++k)                                        \
        C.data()[j*n+i] += A.data()[k*n+i] * (B.data()[k*n+j]);     \
                                                                    \
  DBOUT(std::cerr << CRED "  a*b.'" CRESET "\n");                   \
  D = A*B.transpose();                                              \
  ASSERT_NEAR((double)rpart(D - C).norm(), 0, rpart(C).norm() * tol) << "r|a*b.'"; \
  ASSERT_NEAR((double)dpart(D - C).norm(), 0, dpart(C).norm() * tol) << "d|a*b.'"; \
                                                                    \
  C.setZero();                                                      \
  for(int i=0; i<n; ++i)                                            \
    for(int j=0; j<n; ++j)                                          \
      for(int k=0; k<n; ++k)                                        \
        C.data()[j*n+i] += (A.data()[i*n+k]) * B.data()[j*n+k];     \
                                                                    \
  DBOUT(std::cerr << CRED "  a.'*b'" CRESET "\n");                  \
  D = A.transpose()*B;                                              \
  ASSERT_NEAR((double)rpart(D - C).norm(), 0, rpart(C).norm() * tol) << "r|a.'*b"; \
  ASSERT_NEAR((double)dpart(D - C).norm(), 0, dpart(C).norm() * tol) << "d|a.'*b"; \
                                                                    \
  C.setZero();                                                      \
  for(int i=0; i<n; ++i)                                            \
    for(int j=0; j<n; ++j)                                          \
      for(int k=0; k<n; ++k)                                        \
        C.data()[j*n+i] += conj(A.data()[i*n+k]) * conj(B.data()[k*n+j]); \
                                                                        \
  DBOUT(std::cerr << CRED "  a'*b'" CRESET "\n");                       \
  D = A.adjoint()*B.adjoint();                                          \
  ASSERT_NEAR((double)rpart(D - C).norm(), 0, rpart(C).norm() * tol) << "r|a'*b'"; \
  ASSERT_NEAR((double)dpart(D - C).norm(), 0, dpart(C).norm() * tol) << "d|a'*b'"; \
  }

#if defined(PHASE_1)

MAKE_MULT_TEST(double, double, 65, 65)
MAKE_MULT_TEST(complexf, complexf, 2, 2)
MAKE_MULT_TEST(complexf, complexf, 4, 4)
//MAKE_MULT_TEST(complexf, complexf, 8, 8)
MAKE_MULT_TEST(complexf, complexf, Eigen::Dynamic, 8)
MAKE_MULT_TEST(complexf, complexf, Eigen::Dynamic, 64)
//MAKE_MULT_TEST(complexf, complexf, Eigen::Dynamic, 129)
MAKE_MULT_TEST(complexd, complexd, Eigen::Dynamic, 129)

#elif defined(PHASE_2)

MAKE_MULT_TEST(double, double, 2, 2)
MAKE_MULT_TEST(complexf, float, 2, 2)
MAKE_MULT_TEST(float, complexf, 2, 2)
//MAKE_MULT_TEST(complexf, float, 4, 4)
//MAKE_MULT_TEST(float, complexf, 4, 4)
//MAKE_MULT_TEST(complexf, float, 24, 24)
//MAKE_MULT_TEST(float, complexf, 24, 24)
MAKE_MULT_TEST(complexf, float, Eigen::Dynamic, 31)
MAKE_MULT_TEST(float, complexf, Eigen::Dynamic, 31)
//MAKE_MULT_TEST(complexd, double, Eigen::Dynamic, 31)
//MAKE_MULT_TEST(double, complexd, Eigen::Dynamic, 31)

#elif defined(PHASE_3)

MAKE_MULT_TEST(dualf, dualf, 2, 2)
MAKE_MULT_TEST(dualf, dualf, 3, 3)
MAKE_MULT_TEST(dualf, dualf, 4, 4)
MAKE_MULT_TEST(dualf, dualf, 7, 7)
MAKE_MULT_TEST(dualf, dualf, 8, 8)
MAKE_MULT_TEST(dualf, dualf, 31, 31)
MAKE_MULT_TEST(dualf, dualf, Eigen::Dynamic, 127)
//MAKE_MULT_TEST(dualf, float, 31, 31)
//MAKE_MULT_TEST(float, dualf, 31, 31)

MAKE_MULT_TEST(cdualf, cdualf, 2,2)
MAKE_MULT_TEST(cdualf, cdualf, 3,3)
MAKE_MULT_TEST(cdualf, cdualf, 4,4)
MAKE_MULT_TEST(cdualf, cdualf, 31,31)
MAKE_MULT_TEST(cdualf, cdualf, Eigen::Dynamic, 127)
//MAKE_MULT_TEST(cdualf, cdualf, Eigen::Dynamic, 255)

#elif defined(PHASE_4)

MAKE_MULT_TEST(cdualf, dualf, 4,4)
MAKE_MULT_TEST(dualf, cdualf, 4,4)

MAKE_MULT_TEST(dualf, cdualf, 23,23)
MAKE_MULT_TEST(dualf, cdualf, Eigen::Dynamic,127)

MAKE_MULT_TEST(cdualf, dualf, 8,8)
MAKE_MULT_TEST(cdualf, dualf, Eigen::Dynamic,127)

#elif defined(PHASE_5)

MAKE_MULT_TEST(duald, duald, 2, 2)
MAKE_MULT_TEST(duald, duald, 3, 3)
MAKE_MULT_TEST(duald, duald, 4, 4)
MAKE_MULT_TEST(duald, duald, 31, 31)
MAKE_MULT_TEST(duald, duald, Eigen::Dynamic, 127)
//MAKE_MULT_TEST(duald, duald, Eigen::Dynamic, 255)

MAKE_MULT_TEST(cduald, duald, 4,4)
MAKE_MULT_TEST(duald, cduald, 4,4)

MAKE_MULT_TEST(cduald, cduald, Eigen::Dynamic, 127)

MAKE_MULT_TEST(duald, cduald, Eigen::Dynamic,127)
MAKE_MULT_TEST(cduald, duald, Eigen::Dynamic,127)

#endif

TEST(flags, VECTORIZE) {
#if defined(CPPDUALS_DONT_VECTORIZE) || defined(EIGEN_DONT_VECTORIZE)
  EXPECT_TRUE(false) << "CPPDUALS_DONT_VECTORIZE supresses vectorization tests!";
#endif
}
TEST(flags, SSE) {
  EXPECT_TRUE(std::string(Eigen::SimdInstructionSetsInUse()).find("SSE") != std::string::npos)
    << "Not using SSE instructions:" << Eigen::SimdInstructionSetsInUse();
#ifndef EIGEN_VECTORIZE_SSE
  EXPECT_TRUE(false)
    << "Not using EIGEN_VECTORIZE_SSE:" << Eigen::SimdInstructionSetsInUse();
#endif
}
TEST(flags, AVX) {
  EXPECT_TRUE(std::string(Eigen::SimdInstructionSetsInUse()).find("AVX") != std::string::npos)
    << "Not using AVX instructions:" << Eigen::SimdInstructionSetsInUse();
#ifndef EIGEN_VECTORIZE_AVX
  EXPECT_TRUE(false)
    << "Not using EIGEN_VECTORIZE_AVX:" << Eigen::SimdInstructionSetsInUse();
#endif
}

#define QUOTE(...) STRFY(__VA_ARGS__)
#define STRFY(...) #__VA_ARGS__

int main(int argc, char **argv)
{
  std::ptrdiff_t l1, l2, l3;
  Eigen::internal::manage_caching_sizes(Eigen::GetAction, &l1, &l2, &l3);

  std::cout << "l1=" << l1 << " l2=" << l2 << " l3=" << l3 << "\n";
  std::cout << "OPT_FLAGS=" << QUOTE(OPT_FLAGS) << "\n";
  std::cout << "INSTRUCTIONSET=" << Eigen::SimdInstructionSetsInUse() << "\n";
  ::testing::InitGoogleTest(&argc, argv);
  std::cout.precision(20);
  std::cerr.precision(20);
  return RUN_ALL_TESTS();
}

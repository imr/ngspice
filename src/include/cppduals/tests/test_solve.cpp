//===-- test_funcs.cpp - test duals/dual ------------------------*- C++ -*-===//
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
 * \file test_eigen Dual number Eigen integration tests
 *
 * (c)2019 Michael Tesch. tesch1@gmail.com
 */

#include "type_name.hpp"
#include <duals/dual_eigen>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/StdVector>
#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/AutoDiff>
#include "eexpokit/padm.hpp"
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
#define EXPECT_DEQ(A,B) EXPECT_EQ(rpart(A), rpart(B)); EXPECT_EQ(dpart(A), dpart(B))
#define ASSERT_DEQ(A,B) ASSERT_EQ(rpart(A), rpart(B)); ASSERT_EQ(dpart(A), dpart(B))
#define EXPECT_DNE(A,B) EXPECT_NE(rpart(A), rpart(B)); EXPECT_NE(dpart(A), dpart(B))
#define EXPECT_DNEAR(A,B,tol)                      \
  EXPECT_NEAR(rpart(A), rpart(B),tol);             \
  EXPECT_NEAR(dpart(A), dpart(B),tol)

template <class T, int NN = 30, class DT = dual<T> >
void solveLu() {
  auto tol = NN * NN * 10000 * Eigen::NumTraits<T>::epsilon();

  // check scalar
  emtx<T,NN> a = emtx<T,NN>::Random();
  emtx<T,NN> b = emtx<T,NN>::Random();
  emtx<T,NN> c,d,e;

  c = a * b;
  d = a.lu().solve(c);
  EXPECT_LT((b - d).norm(), tol);

  // check dual
  emtx<DT,NN> A = a + DT(0,1) * emtx<T,NN>::Random();
  emtx<DT,NN> B = b + DT(0,1) * emtx<T,NN>::Random();
  emtx<DT,NN> C,D,E;
  C = A * B;
  D = A.lu().solve(C);
  EXPECT_LT(rpart(B - D).norm(), tol);
  EXPECT_LT(dpart(B - D).norm(), tol);
}

#if defined(PHASE_1)

TEST(solveLu, float2) { solveLu<float,2>(); }
TEST(solveLu, float7) { solveLu<float,7>(); }
TEST(solveLu, float8) { solveLu<float,8>(); }
TEST(solveLu, float31) { solveLu<float,31>(); }

TEST(solveLu, double2) { solveLu<double,2>(); }
TEST(solveLu, double3) { solveLu<double,3>(); }
TEST(solveLu, double4) { solveLu<double,4>(); }
TEST(solveLu, double31) { solveLu<double,31>(); }

#elif defined(PHASE_2)

TEST(solveLu, complexf2) { solveLu<complexf,2,cdualf>(); }
TEST(solveLu, complexf8) { solveLu<complexf,8,cdualf>(); }
TEST(solveLu, complexf31) { solveLu<complexf,31,cdualf>(); }

TEST(solveLu, complexd31) { solveLu<complexd,31,cduald>(); }

#endif

#define QUOTE(...) STRFY(__VA_ARGS__)
#define STRFY(...) #__VA_ARGS__

int main(int argc, char **argv)
{
  std::cout << "OPT_FLAGS=" << QUOTE(OPT_FLAGS) << "\n";
  std::cout << "INSTRUCTIONSET=" << Eigen::SimdInstructionSetsInUse() << "\n";
  ::testing::InitGoogleTest(&argc, argv);
  std::cout.precision(20);
  std::cerr.precision(20);
  return RUN_ALL_TESTS();
}

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

/// Simple taylor series, truncated when |S| is "small enough"
template <class DerivedA, typename ReturnT = typename DerivedA::PlainObject>
ReturnT
expm4(const Eigen::EigenBase<DerivedA> & A_,
      typename DerivedA::RealScalar mn = std::numeric_limits<typename DerivedA::RealScalar>::epsilon() * 1000)
{
  //std::cerr << "do    PO:" << type_name<typename DerivedA::PlainObject>() << "\n";
  typedef typename DerivedA::RealScalar Real;
  using std::isfinite;
  const DerivedA & A = A_.derived();
  int maxt = std::numeric_limits<Real>::digits;
  int s = log2(rpart(A.derived().norm())) + 1;
  s = std::max(0, s);

  auto B = A * pow(Real(2), -s);
  ReturnT R(A.rows(), A.cols());
  R.setIdentity();
  R += B;
  ReturnT S = B;
  for (int ii = 2; ii < maxt; ii++) {
    S = S * B * Real(1.0/ii);
    R += S;
    auto Sn = S.norm();
    if (!isfinite(Sn)) {
      std::cout << "expm() non-finite norm:" << Sn << " at " << ii << "\n";
      std::cout << " |R| = " << R.norm() << " s=" << s << "\n"
                << " |A| = " << rpart(A.real().norm()) << "\n"
                << " |A/2^s|=" << rpart(A.real().norm()/pow(2,s)) << "\n";
      break;
    }
    // converged yet?
    if (Sn < mn)
      break;
    if (ii == maxt - 1) {
      std::cout << "expm() didn't converge in " << maxt << " |S| = " << Sn << "\n";
      throw std::invalid_argument("no converge");
    }
  }

  for (; s; s--)
    R = R * R;
  return R;
}

template <class T, int NN = 30, class DT = dual<T> >
void dexpm() {
  //typedef std::complex<float> T;
  //typedef std::complex<dual<T>> dualt;
  auto tol = NN * NN * 10000 * Eigen::NumTraits<T>::epsilon();

#define N2 2*NN

  // check dual
  emtx<T,NN> A = emtx<T,NN>::Random();
  emtx<T,NN> V = emtx<T,NN>::Random();
  emtx<T,NN> dA1,dA2,dA3,eA1,eA2,eA3;

  // dA/dV method 1
  emtx<T,N2> AVA = emtx<T,N2>::Zero();
  AVA.block( 0, 0,NN,NN) = A;
  AVA.block( 0,NN,NN,NN) = V;
  AVA.block(NN,NN,NN,NN) = A;
  AVA = AVA.exp();
  eA1 = AVA.block(0,0,NN,NN);
  dA1 = AVA.block(0,NN,NN,NN);

  // dA/dV method 2
  emtx<DT, NN> a,b;
  a = A + DT(1_e) * V;
  b = eexpokit::padm(a,13);
  //b = expm4(a);
  //b = a.exp();
  eA2 = rpart(b);
  dA2 = dpart(b);

#if 0
  // dA/dV method 3
  //typedef Eigen::AutoDiffScalar<emtx<T,NN>> AD;
  typedef Eigen::Matrix<T,1,1> Vector1t;
  typedef Eigen::AutoDiffScalar<Vector1t> AD;
  typedef Eigen::Matrix<AD,NN,NN> MatrixAD;
  //MatrixAD c = A.template cast<AD>();
  MatrixAD c(A);
  for (Eigen::Index j = 0; j < NN; j++)
    for (Eigen::Index i = 0; i < NN; i++)
      c(i,j).derivatives()(0) = V(i,j);
  //auto d = eexpokit::padm(c,8);
  MatrixAD d = c.exp();
  eA3 = d;
  for (Eigen::Index j = 0; j < NN; j++)
    for (Eigen::Index i = 0; i < NN; i++)
      dA3(i,j) = d(i,j).derivatives()(0);

  EXPECT_LT((dA1 - dA3).norm(), tol) << "dA1=" << dA1.block(0,0,std::min(4,NN),std::min(4,NN)) << "\n"
                                     << "dA3=" << dA3.block(0,0,std::min(4,NN),std::min(4,NN)) << "\n";
#endif

#if 0
  std::cout << "A=" << A << "\n";
  std::cout << "V=" << V << "\n";
  std::cout << "a=" << a << "\n";
  std::cout << "b=" << b << "\n";
#endif
  EXPECT_LT((eA1 - eA2).norm(), tol) << "eA1=" << eA1.block(0,0,std::min(4,NN),std::min(4,NN)) << "\n"
                                     << "eA2=" << eA2.block(0,0,std::min(4,NN),std::min(4,NN)) << "\n";
  EXPECT_LT((dA1 - dA2).norm(), tol) << "dA1=" << dA1.block(0,0,std::min(4,NN),std::min(4,NN)) << "\n"
                                     << "dA2=" << dA2.block(0,0,std::min(4,NN),std::min(4,NN)) << "\n";
#undef NN
#undef N2
}

#if defined(PHASE_1)

TEST(dexpm, float2) { dexpm<float,2>(); }
//TEST(dexpm, float3) { dexpm<float,3>(); }
TEST(dexpm, float7) { dexpm<float,7>(); }

#elif defined(PHASE_2)

TEST(dexpm, float8) { dexpm<float,8>(); }
//TEST(dexpm, float16) { dexpm<float,16>(); }
TEST(dexpm, float31) { dexpm<float,31>(); }

#elif defined(PHASE_3)

TEST(dexpm, double2) { dexpm<double,2>(); }
TEST(dexpm, double3) { dexpm<double,3>(); }

#elif defined(PHASE_4)

TEST(dexpm, double4) { dexpm<double,4>(); }
TEST(dexpm, double31) { dexpm<double,31>(); }
//TEST(dexpm, adouble) { dexpm<double, adouble>(); }

#elif defined(PHASE_5)

TEST(dexpm, cfloat8) { dexpm<complexf,8,cdualf>(); }
TEST(dexpm, cfloat31) { dexpm<complexf,31,cdualf>(); }

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

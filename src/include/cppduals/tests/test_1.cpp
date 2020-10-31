//===-- test_1.cpp - test duals/dual ------------------------*- C++ -*-===//
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
#include <fstream>
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

#define QUOTE(...) STRFY(__VA_ARGS__)
#define STRFY(...) #__VA_ARGS__

/// Simple taylor series, truncated when |S| is "small enough"
template <class DerivedA, typename ReturnT = typename DerivedA::PlainObject>
ReturnT
expm4(const Eigen::EigenBase<DerivedA> & A_,
      typename DerivedA::RealScalar mn
      = std::numeric_limits<typename DerivedA::RealScalar>::epsilon() * 10
      //= 1.L/(1ull << (std::numeric_limits<typename DerivedA::RealScalar>::digits / 3))
      )
{
  //std::cerr << "do    PO:" << type_name<typename DerivedA::PlainObject>() << "\n";
  typedef typename DerivedA::RealScalar Real;
  using std::isfinite;
  using std::pow;
  const DerivedA & A = A_.derived();
  int maxt = std::numeric_limits<Real>::digits;
  int s = (int)log2(rpart(A.derived().norm())) + 1;
  s = std::max(0, s);

  auto B = A * pow(Real(2), -s);
  ReturnT R(A.rows(), A.cols());
  R.setIdentity();
  R += B;
  ReturnT S = B;
  int ni = 0;
  for (int ii = 2; ii < maxt; ii++) {
    ni++;
    S = Real(1.0/ii) * S * B;
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

  //int ss = s;
  for (; s; s--)
    R = R * R;
  //std::cout << "expm s=" << ss << " ni=" << ni << " tol=" << mn << " |S|= " << S.norm() << "\n";
  return R;
}

template <class T, int NN = 30, class DT = dual<T> >
void dexpm() {
  //typedef std::complex<float> T;
  //typedef std::complex<dual<T>> dualt;
  T tol = NN * NN * 1000 * Eigen::NumTraits<T>::epsilon();

#define N2 2*NN

  // check dual
  emtx<T,NN> A = emtx<T,NN>::Random();
  emtx<T,NN> V = emtx<T,NN>::Random();
  emtx<T,NN> dA1,dA2,dA3,eA1,eA2,eA3,C;

  // dA/dV method 1
  emtx<T,N2> AVA = emtx<T,N2>::Zero();
  AVA.block( 0, 0,NN,NN) = A;
  AVA.block( 0,NN,NN,NN) = V;
  AVA.block(NN,NN,NN,NN) = A;
  AVA = AVA.exp();
  //AVA = eexpokit::padm(AVA,13);
  eA1 = AVA.block(0,0,NN,NN);
  dA1 = AVA.block(0,NN,NN,NN);

  // dA/dV method 2
  emtx<DT, NN> a,b,c;
  a = A + DT(0,1) * V;
  b = expm4(a);
  // TODO: find a second trustworthy expm function, padm() and a.exp()
  // arent reliable here.
  //c = eexpokit::padm(a,13);
  //C = eexpokit::padm(A,13);
  //c = a.exp();
  eA2 = rpart(b);
  dA2 = dpart(b);

  eA3 = rpart(c);
  dA3 = dpart(c);
#if 0
  std::ofstream A_(type_name<T>().str() + std::to_string(NN) + "A.dat");
  A_ << A;
  std::ofstream V_(type_name<T>().str() + std::to_string(NN) + "V.dat");
  V_ << V;

  std::ofstream _(type_name<T>().str() + std::to_string(NN) + "_.dat");
  _ << "A=" << A << "\n";
  _ << "V=" << V << "\n";
  _ << "eA1=" << eA1 << "\n";
  _ << "dA1=" << dA1 << "\n";
  //_ << "a=" << a << "\n";
  _ << "b=" << b << "\n";
  _ << "c=" << c << "\n";
  _ << "C=" << C << "\n";

  std::cout << "a:" << type_name<decltype(a)>() << "\n";
#endif
  EXPECT_LT((eA1 - eA2).norm(), tol) << "eA1=" << eA1.block(0,0,std::min(4,NN),std::min(4,NN)) << "\n"
                                     << "eA2=" << eA2.block(0,0,std::min(4,NN),std::min(4,NN)) << "\n"
                                     << "b=" << b.block(0,0,std::min(4,NN),std::min(4,NN)) << "\n";
  EXPECT_LT((dA1 - dA2).norm(), tol) << "dA1=" << dA1.block(0,0,std::min(4,NN),std::min(4,NN)) << "\n"
                                     << "dA2=" << dA2.block(0,0,std::min(4,NN),std::min(4,NN)) << "\n"
                                     << "b=" << b.block(0,0,std::min(4,NN),std::min(4,NN)) << "\n";
#if 0 // see above.
  EXPECT_LT((eA1 - eA3).norm(), tol) << "eA1=" << eA1.block(0,0,std::min(4,NN),std::min(4,NN)) << "\n"
                                     << "eA3=" << eA3.block(0,0,std::min(4,NN),std::min(4,NN)) << "\n"
                                     << "c=" << c.block(0,0,std::min(4,NN),std::min(4,NN)) << "\n";
  EXPECT_LT((dA1 - dA3).norm(), tol) << "dA1=" << dA1.block(0,0,std::min(4,NN),std::min(4,NN)) << "\n"
                                     << "dA3=" << dA3.block(0,0,std::min(4,NN),std::min(4,NN)) << "\n"
                                     << "c=" << c.block(0,0,std::min(4,NN),std::min(4,NN)) << "\n";
#endif
}

#if defined(PHASE_1)

TEST(dexpm, float2) { dexpm<float,2>(); }
TEST(dexpm, float4) { dexpm<float,4>(); }

#elif defined(PHASE_2)

TEST(dexpm, double2) { dexpm<double,2>(); }
TEST(dexpm, double4) { dexpm<double,4>(); }

#endif

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

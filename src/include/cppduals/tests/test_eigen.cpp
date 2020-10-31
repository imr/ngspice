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
//#include <unsupported/Eigen/AutoDiff>
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

TEST(Eigen, NumTraits) {
  //::testing::StaticAssertTypeEq<Eigen::NumTraits<duals::dual<float>>::Real, float>();
  _EXPECT_TRUE(std::is_same<typename Eigen::NumTraits<duals::dual<float>>::Real, dualf>);
  _EXPECT_TRUE(std::is_same<typename Eigen::NumTraits<cdualf>::Real, dualf>);

  EXPECT_EQ(Eigen::NumTraits<duals::dual<float>>::dummy_precision(),
            Eigen::NumTraits<float>::dummy_precision());
  EXPECT_EQ(Eigen::NumTraits<duals::dual<float>>::digits10(),
            Eigen::NumTraits<float>::digits10());

  EXPECT_EQ(Eigen::NumTraits<cdualf>::dummy_precision(),
            Eigen::NumTraits<float>::dummy_precision());
}

TEST(Eigen, construction)
{
  emtx<double> ed;
  emtx<float> ef;
  emtx<complexd> ecd;
  emtx<complexf> ecf;
  emtx<duald> edd;
  emtx<dualf> edf_;
  emtx<cduald> ecdd;
  emtx<cdualf> ecdf_;
}

TEST(Eigen, sparse)
{
  smtx<double> ed;
  smtx<float> ef;
  smtx<complexd> ecd;
  smtx<complexf> ecf;
  smtx<duald> edd;
  smtx<dualf> edf_;
  smtx<cduald> ecdd;
  smtx<cdualf> ecdf_;
}

TEST(Eigen, expr_type_dense) {
  emtx<double> ed;
  emtx<float> ef;
  emtx<complexd> ecd;
  emtx<complexf> ecf;
  emtx<duald> edd;
  emtx<dualf> edf_;
  emtx<cduald> ecdd;
  emtx<cdualf> ecdf_;

  //_EXPECT_TRUE(std::is_base_of<Eigen::EigenBase<U>,U >::value);

  //_EXPECT_TRUE(std::is_same<std::common_type<emtx<dualf>,dualf>::type, emtx<dualf>>);
  ecf = ecf * 1;
#define PO_EXPR_TYPE(...) typename std::decay<decltype( __VA_ARGS__ )>::type::PlainObject
  _EXPECT_TRUE(std::is_same<PO_EXPR_TYPE(ecf*1), emtx<complexf>>);
  _EXPECT_TRUE(std::is_same<PO_EXPR_TYPE(ecf + ecf), emtx<complexf>>);
  //_EXPECT_TRUE(std::is_same<PO_EXPR_TYPE(ecf * edf_), emtx<cdualf>>); [1]
  //_EXPECT_TRUE(std::is_same<PO_EXPR_TYPE(ecf * 1_ef), emtx<cdualf>>); [1]
  _EXPECT_TRUE(std::is_same<PO_EXPR_TYPE(ecf * cdualf(1,2)), emtx<cdualf>>);
  _EXPECT_TRUE(std::is_same<PO_EXPR_TYPE(edf_ * 2), emtx<dualf>>);
  _EXPECT_TRUE(std::is_same<PO_EXPR_TYPE(edf_ * cdualf(2,2)), emtx<cdualf>>);
  static_assert(!duals::can_promote<dualf, emtx<dualf>>::value, "nope");
  _EXPECT_TRUE(std::is_same<PO_EXPR_TYPE(edf_ * 2_ef), emtx<dualf>>);
  //_EXPECT_TRUE(std::is_same<PO_EXPR_TYPE(edf_ * 2_e), emtx<duald>>);
  _EXPECT_TRUE(std::is_same<PO_EXPR_TYPE(ecf * complexf(1,2)), emtx<complexf>>);
  _EXPECT_TRUE(std::is_same<PO_EXPR_TYPE(ecdf_ * 1), emtx<cdualf>>);
  _EXPECT_TRUE(std::is_same<PO_EXPR_TYPE(ecdf_ * 2_ef), emtx<cdualf>>);
  _EXPECT_TRUE(std::is_same<PO_EXPR_TYPE(ecdf_ * complexf(1,2)), emtx<cdualf>>);
  _EXPECT_TRUE(std::is_same<PO_EXPR_TYPE(ecdf_ * cdualf(1_ef,2_ef)), emtx<cdualf>>);
  _EXPECT_TRUE(std::is_same<PO_EXPR_TYPE(edd * 1_e), emtx<duald>>);
  _EXPECT_TRUE(std::is_same<PO_EXPR_TYPE(edd * 1_ef), emtx<duald>>);
  _EXPECT_TRUE(std::is_same<PO_EXPR_TYPE(ed*1), emtx<double>>);

  auto a1 = ecf * 1;
  auto a2 = ecf * ecf;
  //auto a3 = ecf * edf_; [1]
  //auto a4 = ecf * 1_ef; [1]
  auto a5 = ecf * cdualf(1,2);
  auto a6 = edf_ * 2;
  auto a7 = edf_ * 2_ef;
  auto a8 = ecf * complexf(1,2);
  auto a9 = ecdf_ * 1;
  ecdf_ = ecdf_ * complexf(1,2);
  ecdf_ = ecdf_ * cdualf(1_ef,2_ef);
  auto a12 = edd * 1_e;
  auto a13 = ed * 1;
  auto A1 = a1.eval(); A1 += A1;
  auto A2 = a2.eval(); A2 += A2;
  //auto A3 = a3.eval(); A3 += A3; [1]
  //auto A4 = a4.eval(); A4 += A4; [1]
  auto A5 = a5.eval(); A5 += A5;
  auto A6 = a6.eval(); A6 += A6;
  auto A7 = a7.eval(); A7 += A7;
  auto A8 = a8.eval(); A8 += A8;
  auto A9 = a9.eval(); A9 += A9;
  auto A12 = a12.eval(); A12 += A12;
  auto A13 = a13.eval(); A13 += A13;
}

TEST(init, Constant) {
  edf<3> a = edf<3>::Constant(3,3, 5 + 8_ef);
  ecdf<3> b = ecdf<3>::Constant(3,3, cdualf(4,2 + 3_ef));

  EXPECT_EQ(a(1,1), 5 + 8_ef);
  EXPECT_EQ(b(0,1), cdualf(4,2 + 3_ef));
  EXPECT_TRUE((a.array() < 6).all());
  EXPECT_FALSE((a.array() != 5 + 8_ef).any());
  EXPECT_EQ((a.array() == 5 + 8_ef).count(), 9);
}

TEST(init, setIdentity) {
  edf<3> a = edf<3>::Constant(3,3, 5 + 8_ef);
  ecdf<3> b = ecdf<3>::Constant(3,3, cdualf(4,2 + 3_ef));
  a.setIdentity();
  b.setIdentity();

  EXPECT_EQ(a(0,0), 1);
  EXPECT_EQ(a(0,1), 0);
  EXPECT_EQ(a(0,2), 0);
  EXPECT_EQ(a(1,0), 0);
  EXPECT_EQ(a(1,1), 1);
  EXPECT_EQ(a(1,2), 0);
  EXPECT_EQ(a(2,0), 0);
  EXPECT_EQ(a(2,1), 0);
  EXPECT_EQ(a(2,2), 1);

  EXPECT_EQ(b(0,0).real(), 1);
  EXPECT_EQ(b(0,1).real(), 0);
  EXPECT_EQ(b(0,2).real(), 0);
  EXPECT_EQ(b(1,0).real(), 0);
  EXPECT_EQ(b(1,1).real(), 1);
  EXPECT_EQ(b(1,2).real(), 0);
  EXPECT_EQ(b(2,0).real(), 0);
  EXPECT_EQ(b(2,1).real(), 0);
  EXPECT_EQ(b(2,2).real(), 1);
}

TEST(init, setZero) {
  edf<3> a = edf<3>::Constant(3,3, 5 + 8_ef);
  ecdf<3> b = ecdf<3>::Constant(3,3, cdualf(4,2 + 3_ef));

  EXPECT_EQ(a(1,1), 5 + 8_ef);

  a.setZero();
  b.setZero();

  EXPECT_EQ(a(0,0), 0);
  EXPECT_EQ(a(0,1), 0);
  EXPECT_EQ(a(0,2), 0);
  EXPECT_EQ(a(1,0), 0);
  EXPECT_EQ(a(1,1), 0);
  EXPECT_EQ(a(1,2), 0);
  EXPECT_EQ(a(2,0), 0);
  EXPECT_EQ(a(2,1), 0);
  EXPECT_EQ(a(2,2), 0);

  EXPECT_EQ(b(0,0).real(), 0);
  EXPECT_EQ(b(0,1).real(), 0);
  EXPECT_EQ(b(0,2).real(), 0);
  EXPECT_EQ(b(1,0).real(), 0);
  EXPECT_EQ(b(1,1).real(), 0);
  EXPECT_EQ(b(1,2).real(), 0);
  EXPECT_EQ(b(2,0).real(), 0);
  EXPECT_EQ(b(2,1).real(), 0);
  EXPECT_EQ(b(2,2).real(), 0);
  EXPECT_EQ(b(0,0).imag(), 0);
  EXPECT_EQ(b(0,1).imag(), 0);
  EXPECT_EQ(b(0,2).imag(), 0);
  EXPECT_EQ(b(1,0).imag(), 0);
  EXPECT_EQ(b(1,1).imag(), 0);
  EXPECT_EQ(b(1,2).imag(), 0);
  EXPECT_EQ(b(2,0).imag(), 0);
  EXPECT_EQ(b(2,1).imag(), 0);
  EXPECT_EQ(b(2,2).imag(), 0);
}

TEST(init, LinSpaced) {
  ecf<3,1> a = ecf<3,1>::LinSpaced(3, 4, complexf(6,7));
  EXPECT_EQ(a(0).real(), 4);
  EXPECT_EQ(a(1).real(), 5);
  EXPECT_EQ(a(2).real(), 6);

  edf<3,1> b = edf<3,1>::LinSpaced(3, 4, 6 + 6_ef);
  EXPECT_EQ(b(0), 4);
  EXPECT_EQ(b(1), 5 + 3_ef);
  EXPECT_EQ(b(2), 6 + 6_ef);
}

TEST(init, Random) {
  complexf c = emtx<complexf,1>::Random()(0,0);
  EXPECT_NE(c.real(), 0);
  EXPECT_NE(c.imag(), 0);

  typedef dualf Rt;
  using duals::random;

  Rt b = emtx<Rt,1>::Random()(0,0);
  EXPECT_NE(b.rpart(), 0);
  //EXPECT_NE(b.dpart(), 0); //

  edf<3> j = edf<3>::Random(3,3);
  EXPECT_NE(j(1,2).rpart(), 0);
  //EXPECT_NE(j(2,0).dpart(), 0); //

  ecdf<3> k = ecdf<3>::Random(3,3);
  EXPECT_NE(k(0,1).real().rpart(), 0);
  //EXPECT_NE(k(1,1).imag().dpart(), 0); //

  using duals::randos::random2;
  dualf x = random2<dualf>();
  EXPECT_NE(x.rpart(), 0);
  EXPECT_NE(x.dpart(), 0);

  duald y = random2<duald>();
  EXPECT_NE(y.rpart(), 0);
  EXPECT_NE(y.dpart(), 0);

  //EXPECT_NE(k(1,1).imag().dpart(), 0); //
}

TEST(assign, ops) {
  emtx<double> ed;
  emtx<float> ef;
  emtx<complexd> ecd;
  emtx<complexf> ecf;
  emtx<duald> edd;
  emtx<dualf> edf_;
  emtx<cduald> ecdd;
  emtx<cdualf> ecdf_;

  ed *= 2;
  ef *= 2;

  //ecd *= 2;
  //ecf *= 2;
  //ecd *= 2_e;
  //ecf *= 2_ef;

  edd *= 2;
  edf_ *= 2;
  edd *= 2_e;
  edf_ *= 2_ef;
  //ecdd *= 2;
  //  ecdf_ *= 2;
}

TEST(Array, math) {
  int N = 5;

  emtx<float> x(N,1);
  emtx<float> y(N,1);
  emtx<float> z(N,1);

  x.array() += 1;
  x += y + z;
  x *= 2;

  emtx<duald> a(N,1);
  emtx<duald> A(N,1);
  emtx<duald> c(N,1);
  a.array() = 1 + 1_e;
  a.array() += 3_e;
  a.array() = 0;
  a.array() += 1;
  A.array() += 2;
  c = a + A;
  c = a * 2;
  c = a * 2_e;
  c = 2 * a;
  c = (3 + 2_e) * a;
  c *= 3;
  c *= 3_e;
  EXPECT_EQ(c(N-1), 27_e);
}

TEST(access, CwiseRpartOp) {
  // on float matrices (pass-through)
  emtx<float,3> a, b = emtx<float,3>::Random();
  a = b.unaryExpr(duals::CwiseRpartOp<float>());
  EXPECT_NE(a.norm(), 0);

  // on dual<float> matrices
  emtx<dualf,3> d;
  a = d.unaryExpr(duals::CwiseRpartOp<dualf>());
  a = d.unaryExpr(duals::CwiseDpartOp<dualf>());
  a = rpart(d);
  a = dpart(d);

  // on complex<dual<float>> matrices
  emtx<complexf,3> g, f = emtx<complexf,3>::Random();
  emtx<cdualf,3> e;
  e.array() = f.array().cast<cdualf>() - 1_ef * f.array().cast<cdualf>();
  g = e.unaryExpr(duals::CwiseRpartOp<cdualf>());
  EXPECT_EQ((f-g).norm(), 0);
  g = f.unaryExpr(duals::CwiseRpartOp<complexf>());
  EXPECT_EQ((f-g).norm(), 0);
  g = rpart(f);
  EXPECT_EQ((f-g).norm(), 0);

  g = e.unaryExpr(duals::CwiseDpartOp<cdualf>());
  EXPECT_EQ((f+g).norm(), 0);
  g = f.unaryExpr(duals::CwiseDpartOp<complexf>());
  EXPECT_EQ((g).norm(), 0);
  g = dpart(f);
  EXPECT_EQ((g).norm(), 0);
}

TEST(access, CwiseDpartOp) {
  // on float matrices (pass-through)
  emtx<float,3> a, b = emtx<float,3>::Random();
  a = b.unaryExpr(duals::CwiseDpartOp<float>());
  EXPECT_EQ(a.norm(), 0);

  // on dual<float> matrices
  emtx<dualf,3> d = b - 1_ef * b;
  a = d.unaryExpr(duals::CwiseDpartOp<dualf>());
  EXPECT_NE(a.norm(), 0);
  EXPECT_EQ((a+b).norm(), 0);

  // on complex<dual<float>> matrices
  emtx<complexf,3> g, f = emtx<complexf,3>::Random();
  emtx<cdualf,3> e;
  e.array() = f.array().cast<cdualf>() - 1_ef * f.array().cast<cdualf>();

  g = e.unaryExpr(duals::CwiseDpartOp<cdualf>());
  EXPECT_EQ((g+f).norm(), 0);
  d = e.real();
  EXPECT_NE(d.norm(), 0);
}

TEST(measure, norm) {
  typedef emtx<complexf, 3> MatrixC;
  MatrixC c = (MatrixC() << 1,2,3, 4,5,6, 7,8,9).finished();

  EXPECT_EQ(c.cwiseAbs().colwise().sum().maxCoeff(), complexf(18));
  //EXPECT_EQ(c.maxCoeff(), complexf(9));

  //typedef duald Rt;
  typedef dualf Rt;
  //typedef cdualf Rt;
  typedef emtx<Rt, 3> MatrixD;
  Rt b = 1+0_ef;
  b = 3;
  Rt d(1);
  MatrixD x;
  x << d;
  MatrixD a = (MatrixD() << 1,2,3, 4,5+5_ef,6, 7,8,9).finished();
  //typename MatrixD::Index index;

  EXPECT_EQ(a.sum(), 45 + 5_ef);
  EXPECT_NEAR(rpart(a.norm()), 16.8819430161341337282, 1e-5);
  EXPECT_NEAR(rpart(a.mean()), 5, 1e-5);
  EXPECT_NEAR(dpart(a.mean()), 0.555555555555555, 1e-5);
  EXPECT_EQ(a.minCoeff(), 1);
  EXPECT_EQ(a.maxCoeff(), 9);
  EXPECT_EQ(a.trace(), 15 + 5_ef);
  EXPECT_EQ(a.squaredNorm(), 285+0_e);
  EXPECT_EQ(a.lpNorm<1>(), 45 + 5_ef);
  //EXPECT_EQ(a.lpNorm<Eigen::Infinity>(), 9);
  // 1-norm
  //EXPECT_EQ(a.colwise().lpNorm<1>().maxCoeff(), 45 + 5_ef);
  EXPECT_EQ(a.cwiseAbs().colwise().sum().maxCoeff(), 18);

}

TEST(dual_decay, stat) {
  emtx<float> ef;
  emtx<complexd> ecd;
  emtx<complexf> ecf;
  emtx<duald> edd;
  emtx<dualf> edf_;
  emtx<cduald> ecdd;
  emtx<cdualf> ecdf_;

  _EXPECT_TRUE(std::is_same<PO_EXPR_TYPE(rpart(ef)), emtx<float>>);
  _EXPECT_TRUE(std::is_same<PO_EXPR_TYPE(dpart(ef)), emtx<float>>);

  _EXPECT_TRUE(std::is_same<PO_EXPR_TYPE(rpart(edf_)), emtx<float>>);
  _EXPECT_TRUE(std::is_same<PO_EXPR_TYPE(dpart(edf_)), emtx<float>>);

  _EXPECT_TRUE(std::is_same<PO_EXPR_TYPE(rpart(ecdf_)), emtx<complexf>>);
  _EXPECT_TRUE(std::is_same<PO_EXPR_TYPE(dpart(ecdf_)), emtx<complexf>>);
}

TEST(dpart, matrix) {
  emtx<float,3> a = emtx<float,3>::Random(3,3);
  emtx<float,3> b = emtx<float,3>::Random(3,3);
  emtx<float,3> c = emtx<float,3>::Random(3,3);
  emtx<float,3> d = emtx<float,3>::Random(3,3);
  emtx<dualf,3> A = a + 1_ef * b;
  emtx<dualf,3> B = c + 1_ef * d;
  emtx<cdualf,3> AA;
  emtx<complexf,3> BB,CC;
  AA.real() = A;
  AA.imag() = B;
  BB.real() = a;
  BB.imag() = c;
  CC.real() = b;
  CC.imag() = d;

  EXPECT_EQ((rpart(A) - a).norm(),0);
  EXPECT_EQ((dpart(A) - b).norm(),0);

  EXPECT_EQ((rpart(AA) - BB).norm(),0);
  EXPECT_EQ((dpart(AA) - CC).norm(),0);
}

TEST(func, expm) {
  typedef float T;
  typedef dual<T> dualt;
  typedef std::complex<dual<T>> cdualt;
#define NN 3
#define N2 6
  emtx<dualt, NN> a,b;
  a = emtx<dualt, NN>::Random();
  a.array() += 1.1 + 2.2_ef;
  a.setZero();
  a = eexpokit::padm(a);
  EXPECT_LT((a - emtx<dualt, NN>::Identity()).norm(), 1e-6) << "a=" << a << "\n";
  a *= 1+2_e;
  EXPECT_LT((a - emtx<dualt, NN>::Identity()).norm(), 1e-6) << "a=" << a << "\n";

  emtx<cdualt, NN> c;
  //b = a + 1_e * emtx<cdualf, 3>::Random();
  c.setZero();
  c = c.exp();
  //c = expm4(c);
  EXPECT_LT((c - emtx<cdualf, NN>::Identity()).norm(), 1e-6) << "b=" << b << "\n";
  #undef NN
  #undef N2
}

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

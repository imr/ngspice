//===-- test_funcs.cpp - test duals/dual ------------------------*- C++ -*-===//
//
// Part of the cppduals Project
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "gtest/gtest.h"
#include <duals/dual>
#include <complex>

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
typedef std::complex<float> complexf;
typedef std::complex<double> complexd;

//using std::complex;

#define _EXPECT_TRUE(...) {typedef __VA_ARGS__ asdf; EXPECT_TRUE(asdf::value); }
#define EXPECT_CNEAR(a,b,prec) EXPECT_NEAR(std::abs((a) - (b)), 0, std::abs(prec))

// rough comparison of a finite-differences approx of the derivative
// with the duals's implementation.  just meant to catch wild wrongness,
// not verify precision.
#define FD_CHECK(T, F, ...)                                             \
  TEST(func##_##T, F) {                                                 \
    for (T x : __VA_ARGS__) {                                           \
      T prec = 100 * std::sqrt(std::numeric_limits<T>::epsilon());      \
      T dd = dpart(F(x + dual<T>(0,1)));                                \
      /*T dx = std::numeric_limits<T>::epsilon() * (T)1000000;      */  \
      T dx = T(1)/ (1ull << (std::numeric_limits<T>::digits / 3));      \
      T fd = (F(x + dx) - F(x - dx)) / (2*dx);                          \
      EXPECT_CNEAR(dd, fd, prec * std::abs(std::max(std::max(dd,fd),T(1)))) \
        << "dd=" << dd << " fd=" << fd << " x=" << x << " dx=" << dx;   \
    }                                                                   \
  }

FD_CHECK(double, exp, {-1,0,1})
FD_CHECK(double, log, {1})
//FD_CHECK(complexd, log, {-1,1}) TODO
FD_CHECK(double, log10, {1})
//FD_CHECK(complexd, log10, {-1,0,1}) TODO
FD_CHECK(double, sqrt, {0.5,1.0})
FD_CHECK(double, cbrt, {-10.,-0.01,0.01,1.0,10.})
//FD_CHECK(complexd, sqrt, {0,1}) TODO
FD_CHECK(double, sin, {-1,0,1})
FD_CHECK(double, cos, {-1,0,1})
FD_CHECK(double, tan, {-1,0,1})
FD_CHECK(double, asin, {-.9,0.,.9})
FD_CHECK(double, acos, {-.9,0.,.9})
FD_CHECK(double, atan, {-10,-1,0,1,10})

// TODO:
//FD_CHECK(double, sinh, {0})
//FD_CHECK(double, cosh, {0})
//FD_CHECK(double, tanh, {0})
//FD_CHECK(double, asinh, {0})
//FD_CHECK(double, acosh, {0})
//FD_CHECK(double, atanh, {0})

FD_CHECK(double, erf, {-1,0,1})
FD_CHECK(double, erfc, {-1,0,1})
FD_CHECK(double, tgamma, {1.,0.5,10.})
FD_CHECK(double, lgamma, {-1.1, 0.5, 1.1, 2.})

// check that functions with poles in their derivatives dont generate
// NaNs at the poles if dpart==0.
#define DZERO_CHECK(F, DZ)                      \
  TEST(zero_##F, DZ) {                          \
    dual<double> d(DZ,0);                       \
    EXPECT_TRUE(std::isfinite(F(d).dpart()));   \
  }

DZERO_CHECK(log, 0)
DZERO_CHECK(sqrt, 0)
DZERO_CHECK(cbrt, 0)
DZERO_CHECK(asin, 1)
DZERO_CHECK(acos, 1)
//DZERO_CHECK(atan, i)
//DZERO_CHECK(atan2, i)

// These dont really cause d/dt = 0, but do a partial check and
// increase code coverage.
DZERO_CHECK(tgamma, 0)
DZERO_CHECK(lgamma, 0)

TEST(func, tgamma) {
  duald x = 10 + 4_e;
  //EXPECT_EQ(tgamma(x).rpart(), 362880); "interestingly", compiling without optimization (-O0) causes this to fail
  EXPECT_NEAR(tgamma(x).rpart(), 362880, 362880 * 100 * std::numeric_limits<double>::epsilon());
}
TEST(func, rpart) {
  dualf x = 10 + 4_e;
  EXPECT_EQ(rpart(x), 10);
}
TEST(func, dpart) {
  dualf x = 2 + 4_e;
  EXPECT_EQ(dpart(x), 4);
}
TEST(func, abs) {
}
TEST(func, arg) {
}
TEST(func, norm) {
}
TEST(func, conj) {
}
TEST(func, polar) {
}

struct pike_f1 {
  // function
  template <typename TYPE>
  TYPE
  f(const TYPE & x) {
    return exp(x) / sqrt(pow(sin(x), 3) + pow(cos(x), 3));
  }

  // analytic derivative
  template <typename TYPE>
  TYPE
  df(const TYPE & x) {
    return (exp(x) * (3 * cos(x) + 5*cos(3*x) + 9 * sin(x) + sin(3*x))) /
      (8 * pow(pow(sin(x), (3)) + pow(cos(x), 3), 3./2.));
  }

  // analytic second derivative
  template <typename TYPE>
  TYPE
  ddf(const TYPE & x) {
    return (exp(x) * (130 - 12 * cos(2*x) + 30*cos(4*x)
                      + 12*cos(6*x)
                      - 111.*sin(2*x) + 48.*sin(4*x) + 5*sin(6*x))) /
      (64. * pow(pow(sin(x), 3) + pow(cos(x), 3), 5./2.));
  }

  // analytic third derivative
  template <typename TYPE>
  TYPE
  dddf(const TYPE & x) {
    return exp(x)*(1.0)
      / pow(sin(x)
            + pow(cos(x),(3.0))
            - pow(cos(x),(2.0)) * sin(x), 7.0/2.0)
      * (cos(x) *
         - (186.0)
         + sin(x)*(68.0)
         + pow(cos(x),3)*171
         - pow(cos(x),5)*42
         - pow(cos(x),7)*33
         + pow(cos(x),9)*110
         + pow(cos(x),2)*sin(x)*(256.0)
         - pow(cos(x),4)*sin(x)*(495.0)
         + pow(cos(x),6)*sin(x)*(139.0)
         + pow(cos(x),8)*sin(x)*74.0) * (1.0/8);
  }
};

TEST(diff, pike) {
#if 1
  typedef double real_t;
  typedef duald dual_t;
  typedef hyperduald hdual_t;
#else
  typedef float real_t;
  typedef dualf dual_t;
  typedef hyperduald hdual_t;
#endif
  pike_f1 f1;
  // calculate f, f' and f'' and f''' analytically at x
  real_t x = 7;
  real_t f = f1.f(x);
  real_t fp = f1.df(x);
  real_t fpp = f1.ddf(x);
  real_t fppp = f1.dddf(x);

  // calculate f, f' and f'' and f'' and f''' using duals
  dual_t dfp = f1.f(x + 1_e);
  dual_t ddfp = f1.df(x + 1_e);
  real_t x4 = 0;
  hdual_t dfpp = f1.f(hdual_t(x+1_e, 1 + x4*1_e) ); // x + 1*e1 + 1*e2 + x4*e1e2
  hdual_t dfppp = f1.df(hdual_t(x+1_e, 1 + x4*1_e));

  real_t prec = std::numeric_limits<real_t>::epsilon() * 1e6;
  //prec = 1e-11;
  EXPECT_NEAR(f, dfpp.rpart().rpart(), prec);
  EXPECT_NEAR(fp, dfp.dpart(), prec);
  EXPECT_NEAR(fp, ddfp.rpart(), prec);
  EXPECT_NEAR(fp, dfpp.rpart().dpart(), prec);
  EXPECT_NEAR(fpp, ddfp.dpart(), prec);
  EXPECT_NEAR(fpp, dfpp.dpart().dpart(), prec);
  EXPECT_NEAR(fppp, dfppp.dpart().dpart(), prec);
  //std::cout << "dfpp=" << dfpp << "\n";
  //std::cout << "dfppp=" << dfppp << "\n";
}

#define QUOTE(...) STRFY(__VA_ARGS__)
#define STRFY(...) #__VA_ARGS__

int main(int argc, char **argv)
{
  std::cout << "OPT_FLAGS=" << QUOTE(OPT_FLAGS) << "\n";
  ::testing::InitGoogleTest(&argc, argv);
  std::cout.precision(20);
  std::cerr.precision(20);
  return RUN_ALL_TESTS();
}

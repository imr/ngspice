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
    using std::isnan;                                                   \
    for (T x : __VA_ARGS__) {                                           \
      T prec = 100 * std::sqrt(std::numeric_limits<T>::epsilon());      \
      T dd = duals::dpart(F(x + dual<T>(0,1)));                         \
      /*T dx = std::numeric_limits<T>::epsilon() * (T)1000000;      */  \
      T dx = T(1)/ (1ull << (std::numeric_limits<T>::digits / 3));      \
      T fd = (F(x + dx) - F(x - dx)) / (2*dx);                          \
      if (!isnan(dd) && !isnan(fd)) {                                   \
        EXPECT_CNEAR(dd, fd,                                            \
                     prec * std::abs(std::max(std::max(dd,fd),T(1))))   \
          << "dd=" << dd << " fd=" << fd << " x=" << x << " dx=" << dx; \
      }                                                                 \
    }                                                                   \
  }

#define powL(x) pow(x,2)
#define powR(x) pow(2,x)
#define powLR(x) pow(x,x)

FD_CHECK(double, exp, {-1,0,1})
FD_CHECK(double, log, {1, 3})
// FD_CHECK(complexd, log, {-1,1}) TODO
FD_CHECK(double, log10, {1, 3})

FD_CHECK(double, powL, {-3.,-1.,-0.4,0.,0.6,1.,2.5})
FD_CHECK(double, powR, {-3.,-1.,-0.4,0.,0.6,1.,2.5})
FD_CHECK(double, powLR, {-3.,-1.,-0.4,0.,0.6,1.,2.5})

// FD_CHECK(complexd, log10, {-1,0,1}) TODO
FD_CHECK(double, sqrt, {0.5,1.0})
FD_CHECK(double, cbrt, {-10.,-0.01,0.01,1.0,10.})
// FD_CHECK(complexd, sqrt, {0,1}) TODO
FD_CHECK(double, sin, {-1,0,1})
FD_CHECK(double, cos, {-1,0,1})
FD_CHECK(double, tan, {-1,0,1})
FD_CHECK(double, asin, {-.9,0.,.9})
FD_CHECK(double, acos, {-.9,0.,.9})
FD_CHECK(double, atan, {-10,-1,0,1,10})

// TODO:
#define atan2L(x) atan2(x,2)
#define atan2R(x) atan2(2,x)
#define atan2LR(x) atan2(x,x)
FD_CHECK(double, atan2L, {-10.,-1.,0.,1.,10.})
FD_CHECK(double, atan2R, {-10.,-1.,0.01,1.,10.})
FD_CHECK(double, atan2LR, {-10.,-1.,0.01,1.,10.})

#define hypot2LR(x) hypot(x,x)
FD_CHECK(double, hypot2LR, {-10.,-1.,0.01,1.,10.})

#define scalbnL(x) scalbn(x,2)
FD_CHECK(double, scalbnL, {-10.,-1.,0.01,1.,10.})

FD_CHECK(double, sinh, {-0.1, 0.1})
FD_CHECK(double, cosh, {-0.1, 0.1})
FD_CHECK(double, tanh, {-0.1, 0.1})
FD_CHECK(double, asinh, {-0.1, 0.1})
FD_CHECK(double, acosh, {-1.1, 1.1})
FD_CHECK(double, atanh, {-0.1, 0.1})
FD_CHECK(double, log1p, {-0.1, 0.1})
FD_CHECK(double, expm1, {-0.1, 0.1})

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

// part selection functions
TEST(func, rpart) {
  dualf x = 10 + 4_e;
  EXPECT_EQ(rpart(x), 10);
}
TEST(func, dpart) {
  dualf x = 2 + 4_e;
  EXPECT_EQ(dpart(x), 4);
}

// non-differentiable operations on the real part.
TEST(func, abs) {
  dualf x = -10 + 4_e;
  EXPECT_EQ(abs(x), 10);
}
TEST(func, fabs) {
  dualf x = -10 + 4_e;
  EXPECT_EQ(fabs(x), 10);
}
TEST(func, fmax) {
  dualf x = -10 + 4_e;
  dualf y = 10 + 4_e;
  EXPECT_EQ(fmax(x, y), 10);
}
TEST(func, fmin) {
  dualf x = -10 + 4_e;
  dualf y = 10 + 4_e;
  EXPECT_EQ(fmin(x, y), -10);
}
TEST(func, frexp) {
  dualf x = 6 + 4_e;
  int exp = 0;
  EXPECT_EQ(frexp(x, &exp), 0.75);
  EXPECT_EQ(exp, 3);
}
TEST(func, ldexp) {
  dualf x = 0.5 + 4_e;
  int exp = 1;
  EXPECT_EQ(ldexp(x, exp), 1);
}
TEST(func, trunc) {
  dualf x = 1.5 + 4_e;
  EXPECT_EQ(trunc(x), 1);
}
TEST(func, floor) {
  dualf x = 1.5 + 4_e;
  EXPECT_EQ(floor(x), 1);
}
TEST(func, ceil) {
  dualf x = 1.5 + 4_e;
  EXPECT_EQ(ceil(x), 2);
}
TEST(func, round) {
  dualf x = 1.5 + 4_e;
  EXPECT_EQ(round(x), 2);
}
// floating point functions
TEST(func, fpclassify) {
  dualf x = 1.5 + 4_e;
  EXPECT_EQ(fpclassify(x), FP_NORMAL);
  EXPECT_EQ(fpclassify(std::numeric_limits<dualf>::infinity()), FP_INFINITE);
  EXPECT_EQ(fpclassify(std::numeric_limits<dualf>::quiet_NaN()), FP_NAN);
  if (std::numeric_limits<dualf>::has_denorm != std::denorm_absent) {
    EXPECT_EQ(fpclassify(std::numeric_limits<dualf>::denorm_min()), FP_SUBNORMAL);
  }
  EXPECT_EQ(fpclassify(x+std::numeric_limits<dualf>::min()), FP_NORMAL);
  EXPECT_EQ(fpclassify(2*std::numeric_limits<dualf>::max()), FP_INFINITE);
  EXPECT_EQ(fpclassify(x+std::numeric_limits<dualf>::epsilon()), FP_NORMAL);
}
TEST(func, isfinite) {
  dualf x = 1.5 + 4_e;
  EXPECT_EQ(isfinite(x), true);
}
TEST(func, isnormal) {
  dualf x = 1.5 + 4_e;
  EXPECT_EQ(isnormal(x), true);
}
TEST(func, isinf) {
  dualf x = 1.5 + 4_e;
  EXPECT_EQ(isinf(x), false);
}
TEST(func, isnan) {
  dualf x = 1.5 + 4_e;
  EXPECT_EQ(isnan(x), false);
}
TEST(func, signbit) {
  dualf x = 1.5 + 4_e;
  EXPECT_EQ(signbit(x), false);
}
TEST(func, copysign) {
  dualf x = 1.5 + 4_e;
  dualf y = -1.3 + 2_e;
  EXPECT_EQ(copysign(x, y), -1.5);
}

// Utility functions
TEST(func, random) {
  dualf x = random(0.001 + 0.001_e, 1 + 1_e);
  EXPECT_GE(rpart(x), 0.001);
  EXPECT_LE(rpart(x), 1);
  EXPECT_GE(dpart(x), 0.001);
  EXPECT_LE(dpart(x), 1);
}

TEST(func, random2) {
  dualf x = duals::randos::random2(0.001 + 0.001_e, 1 + 1_e);
  EXPECT_GE(rpart(x), 0.001);
  EXPECT_LE(rpart(x), 1);
  EXPECT_GE(dpart(x), 0.001);
  EXPECT_LE(dpart(x), 1);
}

// more tests
TEST(func, logb) {
  dualf x = 4 + 1_e;
  EXPECT_EQ(rpart(logb(.5 + 1_e)), -1.);
  EXPECT_EQ(rpart(logb(1 + 1_e)), 0.);
  EXPECT_EQ(rpart(logb(2 + 1_e)), 1.);
  EXPECT_EQ(rpart(logb(3 + 1_e)), 1.);
  EXPECT_EQ(rpart(logb(4 + 1_e)), 2.);
  EXPECT_EQ(rpart(logb(x * x)), 4.);

  EXPECT_EQ(dpart(logb(4 + 8_e)), std::numeric_limits<dualf>::infinity());
  EXPECT_EQ(dpart(logb(4.01 + 8_e)), 0.);

  EXPECT_EQ(dpart(logb(x * x)), std::numeric_limits<dualf>::infinity());
  EXPECT_EQ(dpart(logb(3 * x)), 0.);
  x += 0.01;
  EXPECT_EQ(dpart(logb(3 * x)), 0.);
  EXPECT_EQ(dpart(logb(x * x)), 0.);
}

TEST(func, pow) {
  dualf x = pow(0 + 0_e, 0.);
  EXPECT_EQ(rpart(x), 1);
  EXPECT_EQ(dpart(x), 0);

  dualf y = pow(0 + 0_e, 0. + 0_e);
  EXPECT_EQ(rpart(y), 1);
  EXPECT_EQ(dpart(y), 0);

  dualf z = pow(0, 0. + 0_e);
  EXPECT_EQ(rpart(z), 1);
  EXPECT_EQ(dpart(z), 0);
}
TEST(func, pow_complex) {
  std::complex<dualf> C(3+4_ef, 5+6_ef);
  std::complex<dualf> x = std::pow(C, 2+1_ef);
  std::complex<float> ref = std::complex<float>(std::pow(std::complex<float>(3,5), 2));
  EXPECT_NEAR(rpart(x.real()), ref.real(), 1e-5);
  EXPECT_NEAR(rpart(x.imag()), ref.imag(), 1e-6);
  //EXPECT_EQ(dpart(x), std::pow(std::complex<float>(3, 5), 2) * (2+1_ef));
}

//----------------------------------------------------------------------
// Test for pow_dual_complex(const dual<T>& realBase, const std::complex<dual<T>>& complexExponent)
//----------------------------------------------------------------------
TEST(func, PowDualComplexTest)
{
    // 1) Prepare inputs:
    //    realBase = 2.0 (with zero derivative for this example)
    duald realBase(2.0f);
    
    //    complexExponent = (1.5 + 0.7i),
    //    each part a dual with .rpart() = 1.5 or 0.7, derivative 0
    std::complex<duald> complexExponent(duald(1.5f), duald(0.7f));
    
    // 2) Call the function we want to test:
    //    x^y  =>  pow_dual_complex(realBase, complexExponent)
    std::complex<duald> result = std::pow(realBase, complexExponent);

    // 3) Reference: use standard pow in double
    double dblBase = 2.0;
    std::complex<double> dblExponent(1.5, 0.7);
    std::complex<double> reference = std::pow(dblBase, dblExponent);

    // 4) Compare the .rpart() of the dual’s real/imag with reference
    EXPECT_NEAR(result.real().rpart(), reference.real(), 1e-6);
    EXPECT_NEAR(result.imag().rpart(), reference.imag(), 1e-6);
}

//----------------------------------------------------------------------
// Test for pow_complex_dual(const std::complex<dual<T>>& complexBase, const dual<T>& realExponent)
//----------------------------------------------------------------------
TEST(func, PowComplexDualTest)
{
    // 1) Prepare inputs:
    //    complexBase = (3 + 5i), each part dual with zero derivative
    std::complex<duald> complexBase(duald(3.0f), duald(5.0f));

    //    realExponent = 2.0 as a dual
    duald realExponent(2.0f);

    // 2) Call the function we want to test:
    //    x^y => pow_complex_dual(complexBase, realExponent)
    std::complex<duald> result = pow(complexBase, realExponent);

    // 3) Reference: again, standard pow in double
    std::complex<double> dblBase(3.0, 5.0);
    double               dblExponent = 2.0;
    std::complex<double> reference   = std::pow(dblBase, dblExponent);

    // 4) Compare the .rpart() of the dual’s real/imag with reference
    EXPECT_NEAR(result.real().rpart(), reference.real(), 1e-6);
    EXPECT_NEAR(result.imag().rpart(), reference.imag(), 1e-6);
}
TEST(func, norm) {
  // TODO
}
TEST(func, conj) {
  // TODO
}
TEST(func, polar) {
  // TODO
}
TEST(func, atan) {
  EXPECT_EQ(rpart(atan(0 + 1_e)), atan(0));
  EXPECT_EQ(dpart(atan(1_e)), 1);
  EXPECT_EQ(dpart(atan(1 + 1_e)), 0.5);  // = 1 / (1 + x^2)
  EXPECT_EQ(dpart(atan(-2 + 1_e)), 1. / 5.);  // = 1 / (1 + x^2)
}
TEST(func, atan2) {
  // TODO
  //EXPECT_EQ(dpart(atan2(1_e, 1)), 1);
  //EXPECT_EQ(dpart(atan2(1 + 1_e, 1)), 0.5);
  //EXPECT_EQ(dpart(atan2(-2 + 1_e, 1)), (1. / 5.));
  duald y = 1 + 1_e;
  duald x = 1 + 0_e;
  auto z = atan2(y, x);
  z = atan2(y, x);
  EXPECT_EQ(rpart(z), atan2(rpart(y), rpart(x)));
  EXPECT_EQ(dpart(z), 0.5);

  y = -2 + 1_e;
  x = 1 + 0_e;
  z = atan2(y, x);
  EXPECT_EQ(rpart(z), atan2(rpart(y), rpart(x)));
  EXPECT_EQ(dpart(z), 1. / 5.);

  y = 1 + 0_e;
  x = -2 + 1_e;
  z = atan2(y, x);
  EXPECT_EQ(rpart(z), atan2(rpart(y), rpart(x)));
  EXPECT_EQ(dpart(z), -1. / 5.);
}
TEST(func, atan2a) {
  // TODO
  duald y = 1 + 1_e;
  EXPECT_EQ(rpart(atan2(y, 1)), atan2(rpart(y), 1));
  EXPECT_EQ(rpart(atan2(y, 2)), atan2(rpart(y), 2));
  EXPECT_EQ(dpart(atan2(y, 1)), 0.5);
  y = 2 + 1_e;
  EXPECT_EQ(dpart(atan2(y, -2)), -0.25);
}
TEST(func, atan2b) {
  // TODO
  duald x = 10 + 1_e;
  EXPECT_EQ(rpart(atan2(2, x)), atan2(2, rpart(x)));
  EXPECT_EQ(dpart(atan2(2, x)), -1./52);
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

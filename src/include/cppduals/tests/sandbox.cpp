//===-- test_dual.cpp - test duals/dual -------------------------*- C++ -*-===//
//
// Part of the cppduals Project
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c)2019 Michael Tesch. tesch1@gmail.com
//
/**
 * just a place to play around and get dirty.
 *
 * (c)2019 Michael Tesch. tesch1@gmail.com
 */

#include <duals/dual_eigen>
#include <math.h>
#include <iostream>
#include <iomanip>

#include "type_name.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/MatrixFunctions>

using std::cout;
using std::cerr;
using duals::dualf;
using duals::duald;
using duals::dualf;
using duals::duald;
using duals::dualld;
using duals::hyperdualf;
using duals::hyperduald;
using duals::hyperdualld;
typedef std::complex<double> complexd;
typedef std::complex<float> complexf;
typedef std::complex<duald> cduald;
typedef std::complex<dualf> cdualf;
typedef std::complex<hyperdualf> chyperdualf;

using duals::dual_traits;
using namespace duals::literals;

template <class eT, int N=Eigen::Dynamic, int P=N> using emtx = Eigen::Matrix<eT, N, P>;
template <class eT> using smtx = Eigen::SparseMatrix<eT>;

template <int N=2, int P=N> using ecf = Eigen::Matrix<complexf, N, P> ;
template <int N=2, int P=N> using edf = Eigen::Matrix<dualf, N, P> ;
template <int N=2, int P=N> using ecdf = Eigen::Matrix<cdualf, N, P> ;
#define PO_EXPR_TYPE(...) typename std::decay<decltype( __VA_ARGS__ )>::type::PlainObject


template <class T = int>
class Rando { T x; };

struct Randa { int x; };

namespace std {
template <class T>
struct common_type<Rando<T>,T> { typedef Rando<T> type; };
}

#if EIGEN_ARCH_ARM64

int main(int argc, char * argv[])
{
  emtx<cduald,50> A,B,C;
  //emtx<complexd,50> A,B,C;
  C = A * B;

  //
  using namespace Eigen::internal;
  float32x4_t a, b, v3;
  float32x2_t f2;
  float32x4x2_t yy;
  float ff[] = {1., 2., 3., 4.};
  duals::dual<double> x;
  const uint64x2_t maskq = {0, (uint64_t)-1};
  const uint64x2_t imaskq = {(uint64_t)-1, 0};

  std::cout << "a:" << std::hex << maskq[0] << "\n";
  std::cout << "b:" << maskq[1] << "\n";

  std::cout << "v3=" << v3[0] << ", " << v3[1] << ", " << v3[2] << ", " << v3[3] << "\n";
  std::cout << "f2=" << f2[0] << ", " << f2[1] << "\n";

  std::cout << ff[0] << ", " << ff[1] << ", " << ff[2] << ", " << ff[3] << "\n";
#ifdef __ARM_NEON
  std::cout << "__ARM_NEON:" << __ARM_NEON << "\n";
#endif
#ifdef __ARM_NEON_FP
  std::cout << "__ARM_NEON_FP:" << __ARM_NEON_FP << "\n";
#endif
#ifdef __ARM_ARCH
  std::cout << "__ARM_ARCH:" << __ARM_ARCH << "\n";
#endif
#ifdef __ARM_BIG_ENDIAN
  std::cout << "__ARM_BIG_ENDIAN:" << __ARM_BIG_ENDIAN << "\n";
#endif

  return 0;
}

#elif 0
int main(int argc, char * argv[])
{
  typedef double T;
  T h(T(1) / (1ull << (std::numeric_limits<T>::digits / 3)));
#define func erfc
  for (double x = 0; x < 4; x += .21) {
    std::cout << " erf(" << x << ") = " << func (x) << " , =" << (func (x+h)- func (x))/h << "\n";
    std::cout << "d(erf" << x << ") = " << func (x + 1_e) << "\n";
  }
}

#else

template <class T> T   f(T x) { return pow(x,pow(x,x)); }
template <class T> T  df(T x) { return pow(x,-1 + x + pow(x,x)) * (1 + x*log(x) + x*pow(log(x),2)); }
template <class T> T ddf(T x) { return (pow(x,pow(x,x)) * pow(pow(x,x - 1) + pow(x,x)*log(x)*(log(x) + 1), 2) +
                                        pow(x,pow(x,x)) * (pow(x,x - 1) * log(x) +
                                                           pow(x,x - 1) * (log(x) + 1) +
                                                           pow(x,x - 1) * ((x - 1)/x + log(x)) +
                                                           pow(x,x) * log(x) * pow(log(x) + 1, 2) )); }
int main(int argc, char * argv[])
{
  dualf h;
  dualf xx(1);
  hyperdualf y;
  hyperdualf w(1);
  hyperdualf z(xx,h);
  hyperdualf a(xx);
  emtx<double> ed;
  emtx<float> ef;
  emtx<complexd> ecd;
  emtx<complexf> ecf_;
  emtx<duald> edd;
  emtx<dualf,2> edf_;
  emtx<cduald> ecdd;
  emtx<cdualf> ecdf_;

  std::cout << "  f(2.)            = " << f(2.)    << "\n";
  std::cout << " df(2.)            = " << df(2.)   << "\n";
  std::cout << "ddf(2.)            = " << ddf(2.)  << "\n";
  std::cout << "  f(2+1_e)         = " << f(2+1_e) << "\n";
  std::cout << "  f(2+1_e).dpart() = " << f(2+1_e).dpart() << "\n";

  duals::hyperduald x(2+1_e,1+0_e);
  std::cout << "  f((2+1_e) + (1+0_e)_e).dpart().dpart() = " << f(x).dpart().dpart() << "\n";
  std::cout << "  c((2+1_e) + (1+0_e)_e).dpart().dpart() = " << cbrt(x).dpart().dpart() << "\n";

}

#endif

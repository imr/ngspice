//===-- example.cpp - examples of using <duals/dual> -------*- C++ -*-===//
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
 * \file example.cpp    Dual number usage examples
 *
 * (c)2019 Michael Tesch. tesch1@gmail.com
 */

#if defined(PHASE_1)
#include <duals/dual>

using namespace duals::literals;

template <class T> T   f(T x) { return pow(x,pow(x,x)); }
template <class T> T  df(T x) { return pow(x,-1. + x + pow(x,x)) * (1. + x*log(x) + x*pow(log(x),2.)); }
template <class T> T ddf(T x) { return (pow(x,pow(x,x)) * pow(pow(x,x - 1.) + pow(x,x)*log(x)*(log(x) + 1.), 2.) + 
                                        pow(x,pow(x,x)) * (pow(x,x - 1.) * log(x) + 
                                                           pow(x,x - 1.) * (log(x) + 1.) + 
                                                           pow(x,x - 1.) * ((x - 1.)/x + log(x)) + 
                                                           pow(x,x) * log(x) * pow(log(x) + 1., 2.) )); }

int main()
{
  std::cout << "  f(2.)            = " << f(2.)    << "\n";
  std::cout << " df(2.)            = " << df(2.)   << "\n";
  std::cout << "ddf(2.)            = " << ddf(2.)  << "\n";
  std::cout << "  f(2+1_e)         = " << f(2+1_e) << "\n";
  std::cout << "  f(2+1_e).dpart() = " << f(2+1_e).dpart() << "\n";
  duals::hyperduald x(2+1_e,1+0_e);
  std::cout << "  f((2+1_e) + (1+0_e)_e).dpart().dpart() = " << f(x).dpart().dpart() << "\n";
}

#endif

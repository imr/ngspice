//===-- test_dual.cpp - test duals/dual -------------------------*- C++ -*-===//
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
 * \file test_dual Dual number class tests
 *
 * (c)2019 Michael Tesch. tesch1@gmail.com
 */

#include <duals/dual>
#include <complex>
#include "gtest/gtest.h"

using duals::dualf;
using duals::duald;
using duals::dualld;
using duals::hyperdualf;
using duals::hyperduald;
using duals::hyperdualld;
typedef std::complex<double> complexd;
typedef std::complex<float> complexf;
typedef std::complex<hyperduald> chduald;
typedef std::complex<duald> cduald;
typedef std::complex<dualf> cdualf;
using duals::is_dual;
using duals::is_complex;
using duals::dual_traits;
using namespace duals::literals;

//using std::complex;

#define _EXPECT_TRUE(...)  {typedef __VA_ARGS__ tru; EXPECT_TRUE(tru::value); static_assert(tru::value, "sa"); }
#define _EXPECT_FALSE(...) {typedef __VA_ARGS__ fal; EXPECT_FALSE(fal::value); static_assert(!fal::value, "sa"); }
#define EXPECT_DEQ(A,B) EXPECT_EQ((A).rpart(), (B).rpart()); EXPECT_EQ((A).dpart(), (B).dpart())
#define EXPECT_DNE(A,B) EXPECT_NE((A).rpart(), (B).rpart()); EXPECT_NE((A).dpart(), (B).dpart())

class Rando {};

TEST(template_, dual_traits) {

  // see about maybe using?
  // ::testing::StaticAssertTypeEq<T1, T2>();

  // value_type
  _EXPECT_TRUE(std::is_same<dualf::value_type, float>);
  _EXPECT_TRUE(std::is_same<hyperdualf::value_type, dualf>);

  // real_type
  // depth
  EXPECT_EQ(dual_traits<float>::depth, 0);
  EXPECT_EQ(dual_traits<complexf>::depth, 0);
  EXPECT_EQ(dual_traits<cdualf>::depth, 0);
  EXPECT_EQ(dual_traits<dualf>::depth, 1);
  EXPECT_EQ(dual_traits<hyperdualf>::depth, 2);
}

TEST(template_, external_traits) {
  // is_dual
  // is_arithmetic
  // is_complex
  //
  EXPECT_TRUE(is_dual<dualf>::value);
  EXPECT_TRUE(is_dual<hyperdualf>::value);
  EXPECT_FALSE(is_dual<float>::value);
  EXPECT_FALSE(is_dual<std::complex<float>>::value);

  EXPECT_FALSE(is_complex<dualf>::value);
  EXPECT_FALSE(is_complex<hyperdualf>::value);
  EXPECT_FALSE(is_complex<float>::value);
  EXPECT_FALSE(is_complex<int>::value);
  static_assert(!is_complex<int>::value, "int is complex?");
  static_assert(std::is_same<typename std::enable_if<!duals::is_complex<int>::value,int>::type, int>::value, "yes");

  EXPECT_TRUE(is_complex<std::complex<float>>::value);
  EXPECT_TRUE(is_complex<std::complex<dualf>>::value);

  // is_arithmetic - depends on #define CPPDUALS_ENABLE_STD_IS_ARITHMETIC
  //EXPECT_FALSE(std::is_arithmetic<hyperdualf>::value);
  //EXPECT_FALSE(std::is_arithmetic<Rando>::value);
  EXPECT_FALSE(std::is_arithmetic<std::complex<float>>::value);
  EXPECT_FALSE(std::is_arithmetic<cdualf>::value);
  //EXPECT_TRUE(std::is_arithmetic<dualf>::value);

  EXPECT_TRUE(std::is_compound<complexf>::value);
  EXPECT_TRUE(std::is_compound<cdualf>::value);
  EXPECT_TRUE(std::is_compound<dualf>::value);
}

TEST(numeric_limits, members) {
  // std::numeric_limits
  EXPECT_TRUE(std::numeric_limits<dualf>::is_specialized == true);
  EXPECT_EQ(std::numeric_limits<dualf>::min(), std::numeric_limits<float>::min());
  EXPECT_EQ(std::numeric_limits<dualf>::lowest(), std::numeric_limits<float>::lowest());
  EXPECT_EQ(std::numeric_limits<dualf>::max(), std::numeric_limits<float>::max());
  EXPECT_EQ(std::numeric_limits<dualf>::epsilon(), std::numeric_limits<float>::epsilon());
  EXPECT_EQ(std::numeric_limits<dualf>::round_error(), std::numeric_limits<float>::round_error());
  EXPECT_EQ(std::numeric_limits<dualf>::infinity(), std::numeric_limits<float>::infinity());
  //EXPECT_EQ(std::numeric_limits<dualf>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());
  //EXPECT_EQ(std::numeric_limits<dualf>::signaling_NaN(), std::numeric_limits<float>::signaling_NaN());
  EXPECT_EQ(std::numeric_limits<dualf>::denorm_min(), std::numeric_limits<float>::denorm_min());
}

using namespace duals;
TEST(template_, common_type) {
  _EXPECT_TRUE(std::is_same<std::common_type<int,float>::type, float>);
  _EXPECT_TRUE(std::is_same<std::common_type<dualf,dualf>::type, dualf>);
  _EXPECT_TRUE(std::is_same<std::common_type<dualf,float>::type, dualf>);
  _EXPECT_TRUE(std::is_same<std::common_type<dualf,int>::type, dualf>);
  _EXPECT_TRUE(std::is_same<std::common_type<dualf,double>::type, dualf>); // notable(!) because dual<float>(double)

  _EXPECT_TRUE(std::is_same<std::common_type<float, dualf>::type, dualf>);
  _EXPECT_TRUE(std::is_same<std::common_type<int, dualf>::type, dualf>);
  dualf x;
  hyperdualf y;
  y = true ? x : y;
  _EXPECT_TRUE(std::is_same<duals::promote<hyperdualf,int>::type, hyperdualf>);
  _EXPECT_TRUE(std::is_same<std::common_type<hyperdualf,dualf>::type, hyperdualf>);
  _EXPECT_TRUE(std::is_same<duals::promote<hyperdualf,float>::type, hyperdualf>);
  _EXPECT_TRUE(std::is_same<duals::promote<hyperdualf,hyperduald>::type, hyperduald>);

  _EXPECT_TRUE(std::is_same<duals::promote<int, hyperdualf>::type, hyperdualf>);
  _EXPECT_TRUE(std::is_same<std::common_type<dualf, hyperdualf>::type, hyperdualf>);
  _EXPECT_TRUE(std::is_same<duals::promote<float, hyperdualf>::type, hyperdualf>);
  _EXPECT_TRUE(std::is_same<duals::promote<hyperduald, hyperdualf>::type, hyperduald>);

  _EXPECT_TRUE(std::is_same<std::common_type<std::true_type,float>::type, float>);
  _EXPECT_TRUE(std::is_same<duals::promote<std::true_type,dualf>::type, dualf>);

  // Rando type
  //_EXPECT_FALSE(std::is_same<std::common_type<dualf,Rando>::type, float>); // cant

  // complex
  _EXPECT_TRUE(std::is_same<duals::promote<dualf,complexf>::type, cdualf>);
  _EXPECT_TRUE(std::is_same<duals::promote<complexf,dualf>::type, cdualf>);

  // complex<dual<>>
  _EXPECT_TRUE(std::is_same<std::common_type<cdualf,cdualf>::type, cdualf>);
  _EXPECT_TRUE(std::is_same<std::common_type<dualf,cdualf>::type, cdualf>);
  _EXPECT_TRUE(std::is_same<std::common_type<cdualf,dualf>::type, cdualf>);

  _EXPECT_TRUE(std::is_same<duals::promote<cdualf,int>::type, cdualf>);
  _EXPECT_TRUE(std::is_same<duals::promote<int,cdualf>::type, cdualf>);
  _EXPECT_TRUE(std::is_same<duals::promote<cdualf,float>::type, cdualf>);
  _EXPECT_TRUE(std::is_same<duals::promote<float,cdualf>::type, cdualf>);

  _EXPECT_TRUE(std::is_same<duals::promote<cdualf,complexf>::type, cdualf>);
  _EXPECT_TRUE(std::is_same<duals::promote<complexf,cdualf>::type, cdualf>);

  // expression types
  complexf a;
  dualf d(1,8), dd{2,9};
  hyperduald h(2.f,3.f);
  cdualf cd(1+1_ef,2+2_ef);

  _EXPECT_TRUE(std::is_same<decltype(d*1), dualf>);
  _EXPECT_TRUE(std::is_same<decltype(d*d), dualf>);
  _EXPECT_TRUE(std::is_same<decltype(d*1.f), dualf>);
  _EXPECT_TRUE(std::is_same<decltype(d*1.), duald>);

  _EXPECT_TRUE(std::is_same<decltype(1*d), dualf>);
  _EXPECT_TRUE(std::is_same<decltype(1.f*d), dualf>);

  _EXPECT_TRUE(std::is_same<decltype(y*1), hyperdualf>);
  _EXPECT_TRUE(std::is_same<decltype(y*d), hyperdualf>);
  _EXPECT_TRUE(std::is_same<decltype(y*1.f), hyperdualf>);
  _EXPECT_TRUE(std::is_same<decltype(y*h), hyperduald>);
  _EXPECT_TRUE(std::is_same<decltype(1*y), hyperdualf>);
  _EXPECT_TRUE(std::is_same<decltype(d*y), hyperdualf>);
  _EXPECT_TRUE(std::is_same<decltype(1.f*y), hyperdualf>);
  _EXPECT_TRUE(std::is_same<decltype(h*y), hyperduald>);

  _EXPECT_TRUE(std::is_same<decltype(d*a), cdualf>);
  _EXPECT_TRUE(std::is_same<decltype(a*d), cdualf>);
  _EXPECT_TRUE(std::is_same<decltype(d+a), cdualf>);
  _EXPECT_TRUE(std::is_same<decltype(a+d), cdualf>);

  _EXPECT_TRUE(std::is_same<decltype(cd*cd), cdualf>);
  _EXPECT_TRUE(std::is_same<decltype(cd*d), cdualf>);
  _EXPECT_TRUE(std::is_same<decltype(d*cd), cdualf>);

  _EXPECT_TRUE(std::is_same<decltype(cd*1), cdualf>);
  _EXPECT_TRUE(std::is_same<decltype(1*cd), cdualf>);
  _EXPECT_TRUE(std::is_same<decltype(cd*1.f), cdualf>);
  _EXPECT_TRUE(std::is_same<decltype(1.f*cd), cdualf>);

  _EXPECT_TRUE(std::is_same<decltype(cd*a), cdualf>);
  _EXPECT_TRUE(std::is_same<decltype(a*cd), cdualf>);

  _EXPECT_TRUE(std::is_same<decltype(a + 1_ef), cdualf>);
  _EXPECT_TRUE(std::is_same<decltype(cd * 1), cdualf>);
  _EXPECT_FALSE(duals::can_promote<Rando, int>);
  _EXPECT_FALSE(duals::can_promote<Rando, dualf>);
  //_EXPECT_TRUE(std::is_same<decltype(Rando() + 1_ef), dualf>); // no compile
  _EXPECT_TRUE(std::is_same<decltype(a + hyperduald(2)), chduald>);
  _EXPECT_TRUE(std::is_same<decltype(a * hyperduald(2) + 1_ef), std::complex<hyperduald> > );

  cd = a * dd;
  cd = a + dd;
  cd += d * a;
}

TEST(construction, raw) {
  dualf x;
  dualf y{1};
  dualf yy(1);
  dualf za(2,3);
  dualf zb(7.,9);
  dualf zc(2.,4.);
  dualf w(yy = za);

  duald ze(2.f,3.f);

  hyperdualf X;
  hyperdualf Y(y);
  hyperdualf Z(zb,za);
  hyperdualf W(0,w);

  EXPECT_EQ(x.rpart(), 0);
  EXPECT_EQ(x.dpart(), 0);

  EXPECT_EQ(y.rpart(), 1);
  EXPECT_EQ(y.dpart(), 0);

  EXPECT_EQ(za.rpart(), 2);
  EXPECT_EQ(za.dpart(), 3);

  EXPECT_EQ(zb.rpart(), 7);
  EXPECT_EQ(zb.dpart(), 9);

  EXPECT_EQ(zc.rpart(), 2);
  EXPECT_EQ(zc.dpart(), 4);

  EXPECT_EQ(w.rpart(), 2);
  EXPECT_EQ(w.dpart(), 3);

  EXPECT_EQ(ze.rpart(), 2);
  EXPECT_EQ(ze.dpart(), 3);

  EXPECT_EQ(X.rpart().rpart(), 0);
  EXPECT_EQ(X.rpart().dpart(), 0);
  EXPECT_EQ(X.dpart().rpart(), 0);
  EXPECT_EQ(X.dpart().dpart(), 0);

  EXPECT_EQ(Y.rpart().rpart(), 1);
  EXPECT_EQ(Y.rpart().dpart(), 0);
  EXPECT_EQ(Y.dpart().rpart(), 0);
  EXPECT_EQ(Y.dpart().dpart(), 0);

  EXPECT_EQ(Z.rpart().rpart(), 7);
  EXPECT_EQ(Z.rpart().dpart(), 9);
  EXPECT_EQ(Z.dpart().rpart(), 2);
  EXPECT_EQ(Z.dpart().dpart(), 3);

  EXPECT_EQ(W.rpart().rpart(), 0);
  EXPECT_EQ(W.rpart().dpart(), 0);
  EXPECT_EQ(W.dpart().rpart(), 2);
  EXPECT_EQ(W.dpart().dpart(), 3);
}

TEST(construction, copy) {
  dualf za(2,3);
  dualf zb = za;
  EXPECT_EQ(zb.rpart(), 2);
  EXPECT_EQ(zb.dpart(), 3);
}

#if __cpp_user_defined_literals >= 200809

TEST(literal, expr) {
  dualf x = (2 + 0.0_ef) * (1 + 3_ef);
  duald y = 2.2 * 3.3_e + 0_e;
  dualld z = 4_el / 2 + 0.0_el;

  EXPECT_EQ(x.rpart(), 2);
  EXPECT_EQ(x.dpart(), 6);

  EXPECT_EQ(y.rpart(), 0);
  EXPECT_EQ(y.dpart(), 7.26);

  EXPECT_EQ(z.rpart(), 0);
  EXPECT_EQ(z.dpart(), 2);
}

#endif

TEST(construction, assign) {
  dualf z(2,3);
  EXPECT_EQ(z.dpart(), 3);
  z = 5;
  EXPECT_EQ(z.rpart(), 5);
  EXPECT_EQ(z.dpart(), 0);

  hyperdualf x(1);
  EXPECT_EQ(x.rpart().rpart(), 1);
  EXPECT_EQ(x.rpart().dpart(), 0);
  EXPECT_EQ(x.dpart().rpart(), 0);
  EXPECT_EQ(x.dpart().dpart(), 0);

  x = 4;
  x = 5.0;
  x = 6.0f;
  x = 6.0L;
}

TEST(construction, cast) {
  dualf z = duald(1.1,2);
  EXPECT_EQ(z.dpart(), 2);
  z = z + z;
  int i = (int)z;
  EXPECT_EQ(i,2);
}

TEST(members, rpart) {
  EXPECT_EQ(rpart(3), 3);
  dualf z(2,3);
  EXPECT_EQ(z.rpart(), 2);
  EXPECT_EQ(z.dpart(), 3);
  z.rpart(4);
  EXPECT_EQ(z.rpart(), 4);
  EXPECT_EQ(z.dpart(), 3);
}
TEST(members, dpart) {
  EXPECT_EQ(dpart(3), 0);
  dualf z(2,3);
  EXPECT_EQ(z.dpart(), 3);
  EXPECT_EQ(z.rpart(), 2);
  z.dpart(4);
  EXPECT_EQ(z.dpart(), 4);
  EXPECT_EQ(z.rpart(), 2);
}

TEST(members, unary) {
  dualf z(2,3);
  EXPECT_EQ((-z).rpart(), -2);
  EXPECT_EQ((-z).dpart(), -3);
  EXPECT_EQ((+z).rpart(), 2);
  EXPECT_EQ((+z).dpart(), 3);
}

//TEST(members, real) { TBD
//}

TEST(comparison, eq) {
  dualf a(2,3);
  dualf b(4,1);
  dualf c(2,1);
  dualf d(2,1);
  hyperdualf A(a,a);
  hyperdualf B(b,b);
  hyperdualf C(c,c);
  hyperdualf D(d,d);

  // ==
  EXPECT_TRUE(a == 2);
  EXPECT_TRUE(a == a);
  EXPECT_TRUE(c == d);
  EXPECT_FALSE(a == b);
  EXPECT_TRUE(a == c);
  EXPECT_TRUE(A == 2);
  EXPECT_TRUE(A == A);
  EXPECT_TRUE(C == D);
  EXPECT_FALSE(A == B);
  EXPECT_TRUE(A == C);
  //EXPECT_TRUE(a == 1); //disallowed
  //EXPECT_TRUE(A == 1); //disallowed
  //EXPECT_TRUE(A == a); //disallowed
}

TEST(comparison, ne) {
  dualf a(2,3);
  dualf b(4,1);
  dualf c(2,1);
  dualf d(2,1);
  hyperdualf A(a,a);
  hyperdualf B(b,b);
  hyperdualf C(c,c);
  hyperdualf D(d,d);

  // !=
  EXPECT_TRUE(a != 3);
  EXPECT_FALSE(a != a);
  EXPECT_FALSE(c != d);
  EXPECT_TRUE(a != b);
  EXPECT_FALSE(a != c);
  EXPECT_TRUE(A != 3);
  EXPECT_FALSE(A != A);
  EXPECT_FALSE(C != D);
  EXPECT_TRUE(A != B);
  EXPECT_FALSE(A != C);
  //EXPECT_TRUE(A != a); disallowed
}

TEST(comparison, lt) {
  dualf a(2,3);
  dualf b(4,1);
  dualf c(2,1);
  dualf d(2,1);
  hyperdualf A(a,a);
  hyperdualf B(b,b);
  hyperdualf C(c,c);
  hyperdualf D(d,d);

  // <
  EXPECT_FALSE(a < 2);
  EXPECT_FALSE(a < a);
  EXPECT_FALSE(c < d);
  EXPECT_TRUE(a < b);
  EXPECT_FALSE(a < c);

  EXPECT_FALSE(A < a);
  EXPECT_FALSE(C < d);
  EXPECT_TRUE(A < b);
  EXPECT_FALSE(A < c);
  EXPECT_FALSE(a < A);
  EXPECT_FALSE(c < D);
  EXPECT_TRUE(a < B);
  EXPECT_FALSE(a < C);

  EXPECT_TRUE(a < 2.5);
  EXPECT_TRUE(A < 2.5);
  EXPECT_TRUE(a < 6);
  EXPECT_FALSE(a < 2);
  EXPECT_FALSE(a < 1);

  EXPECT_FALSE(2.5 < a);
  EXPECT_FALSE(6 < a);
  EXPECT_FALSE(2 < a);
  EXPECT_TRUE(1 < a);

}
TEST(comparison, gt) {
  dualf a(2,3);
  dualf b(4,1);
  dualf c(2,1);
  dualf d(2,1);
  hyperdualf A(a,a);
  hyperdualf B(b,b);
  hyperdualf C(c,c);
  hyperdualf D(d,d);

  // >
  EXPECT_FALSE(a > a);
  EXPECT_FALSE(c > d);
  EXPECT_FALSE(a > b);
  EXPECT_FALSE(a > c);
  EXPECT_TRUE(b > c);
  EXPECT_TRUE(b > a);

  EXPECT_FALSE(A > a);
  EXPECT_FALSE(C > d);
  EXPECT_FALSE(A > b);
  EXPECT_FALSE(A > c);
  EXPECT_FALSE(a > A);
  EXPECT_FALSE(c > D);
  EXPECT_FALSE(a > B);
  EXPECT_FALSE(a > C);
  EXPECT_TRUE(B > c);
  EXPECT_TRUE(B > a);
  EXPECT_TRUE(b > C);
  EXPECT_TRUE(b > A);

  EXPECT_FALSE(a > 2.5);
  EXPECT_FALSE(a > 6);
  EXPECT_FALSE(a > 2);
  EXPECT_TRUE(a > 1);
  EXPECT_TRUE(A > 1);

  EXPECT_TRUE(2.5 > a);
  EXPECT_TRUE(6 > a);
  EXPECT_FALSE(2 > a);
  EXPECT_FALSE(1 > a);
  EXPECT_TRUE(3 > A);

}
TEST(comparison, le) {
  dualf a(2,3);
  dualf b(4,1);
  dualf c(2,1);
  dualf d(2,1);
  hyperdualf A(a,a);
  hyperdualf B(b,b);
  hyperdualf C(c,c);
  hyperdualf D(d,d);

  // <=
  EXPECT_TRUE(a <= a);
  EXPECT_TRUE(c <= d);
  EXPECT_TRUE(a <= b);
  EXPECT_TRUE(a <= c);
  EXPECT_FALSE(b <= c);

  EXPECT_TRUE(A <= a);
  EXPECT_TRUE(C <= d);
  EXPECT_FALSE(B <= c);
  EXPECT_TRUE(A <= b);
  EXPECT_TRUE(A <= c);
  EXPECT_TRUE(a <= A);
  EXPECT_TRUE(c <= D);
  EXPECT_TRUE(a <= B);
  EXPECT_TRUE(a <= C);

  EXPECT_TRUE(a <= 2.5);
  EXPECT_TRUE(a <= 6);
  EXPECT_TRUE(a <= 2);
  EXPECT_FALSE(a <= 1);

  EXPECT_FALSE(2.5 <= a);
  EXPECT_FALSE(6 <= a);
  EXPECT_TRUE(2 <= a);
  EXPECT_TRUE(1 <= a);

}
TEST(comparison, ge) {
  dualf a(2,3);
  dualf b(4,1);
  dualf c(2,1);
  dualf d(2,1);
  hyperdualf A(a,a);
  hyperdualf B(b,b);
  hyperdualf C(c,c);
  hyperdualf D(d,d);

  // >=
  EXPECT_TRUE(a >= a);
  EXPECT_TRUE(c >= d);
  EXPECT_FALSE(a >= b);
  EXPECT_TRUE(a >= c);

  EXPECT_TRUE(A >= a);
  EXPECT_TRUE(C >= d);
  EXPECT_FALSE(A >= b);
  EXPECT_TRUE(A >= c);
  EXPECT_TRUE(a >= A);
  EXPECT_TRUE(c >= D);
  EXPECT_FALSE(a >= B);
  EXPECT_TRUE(a >= C);

  EXPECT_FALSE(a >= 2.5);
  EXPECT_FALSE(a >= 6);
  EXPECT_TRUE(a >= 2);
  EXPECT_TRUE(a >= 1);

  EXPECT_TRUE(2.5 >= a);
  EXPECT_TRUE(6 >= a);
  EXPECT_TRUE(2 >= a);
  EXPECT_FALSE(1 >= a);
}

TEST(compound_assign, same_type) {
  // OP=
  dualf x = 2 + 4_e;
  x *= x;
  x /= 2 + 4_e;
  x += 3.;
  x += 30 + 22.5_e;
  x -= 33.f;
  x -= 22.5_el;
  EXPECT_EQ(x.rpart(), 2);
  EXPECT_EQ(x.dpart(), 4);

  x += 1;
  x += 1.2;
  x += 1.2_e;

  x -= 1;
  x -= 1.2;
  x -= 1.2_e;

  x *= 1;
  x *= 1.2;
  x *= 1.2_e;

  x /= 1;
  x /= 1.2;
  x /= 1.2_e;
}

TEST(compound_assign, other_type) {
  dualf x = 2 + 4_e;
  hyperdualf y = 3 + 5_ef;

  y += x;
  y -= x;
  y *= x;
  y /= x;
  //x /= y; //impossible

  y += 1;
  y += 1.2;
  y += 1.2_e;

  y -= 1;
  y -= 1.2;
  y -= 1.2_e;

  y *= 1;
  y *= 1.2;
  y *= 1.2_e;

  y /= 1;
  y /= 1.2;
  y /= 1.2_e;

}

TEST(non_class, rd_part) {
  float z = 3;
  dualf x = 2 + 4_e;
  cdualf y(2 + 4_ef, 5 + 7_ef);

  EXPECT_EQ(rpart(z), 3);
  EXPECT_EQ(dpart(z), 0);
  EXPECT_EQ(rpart(x), 2);
  EXPECT_EQ(dpart(x), 4);
  EXPECT_EQ(rpart(y), complexf(2,5));
  EXPECT_EQ(dpart(y), complexf(4,7));
}

TEST(non_class, dconj) {
  float z = 3;
  dualf x = 2 + 4_ef;
  cdualf y(2 + 4_ef, 5 + 7_ef);

  EXPECT_EQ(dconj(z), 3);
  EXPECT_DEQ(dconj(x), 2 - 4_ef);
  EXPECT_DEQ(real(dconj(y)), 2 - 4_ef);
  EXPECT_DEQ(imag(dconj(y)), 5 - 7_ef);
}

TEST(non_class, putto) {
  {
    std::stringstream s;
    s << 2 + 3_ef;
    EXPECT_EQ(s.str(), "(2+3_ef)");
  }
  {
    std::stringstream s;
    s << 2 + 3_e;
    EXPECT_EQ(s.str(), "(2+3_e)");
  }
  {
    std::stringstream s;
    s << 2 + 3_el;
    EXPECT_EQ(s.str(), "(2+3_el)");
  }
#if 0 // seems reasonable
  {
    std::stringstream s;
    s << 3_e;
    EXPECT_EQ(s.str(), "(3_e)");
  }
#endif
  {
    std::stringstream s;
    s << std::fixed << std::setprecision(10) << 10 + 2_e;
    EXPECT_EQ(s.str(), "(10.0000000000+2.0000000000_e)");
  }
  {
    std::stringstream s;
    s << cdualf(2 + 3_ef, 4 + 5_ef);
    EXPECT_EQ(s.str(), "((2+3_ef),(4+5_ef))");
  }
}

TEST(non_class, getfro) {
  {
    std::stringstream s("( 2+3_ef )");
    dualf x;
    s >> x;
    EXPECT_TRUE(s);
    EXPECT_EQ(x.rpart(), 2);
    EXPECT_EQ(x.dpart(), 3);
  }
  {
    std::stringstream s("( 2+-3_ef )");
    dualf x;
    s >> x;
    EXPECT_TRUE(s);
    EXPECT_EQ(x.rpart(), 2);
    EXPECT_EQ(x.dpart(), -3);
  }
  {
    std::stringstream s("(+2+-3_ef )");
    dualf x;
    s >> x;
    EXPECT_TRUE(s);
    EXPECT_EQ(x.rpart(), 2);
    EXPECT_EQ(x.dpart(), -3);
  }
  {
    std::stringstream s("(2 -3.3_e ) ");
    duald x;
    s >> x;
    EXPECT_TRUE(s);
    EXPECT_EQ(x.rpart(), 2);
    EXPECT_EQ(x.dpart(), -3.3);
  }
  {
    std::stringstream s("(2) ");
    duald x(1,8);
    s >> x;
    EXPECT_TRUE(s);
    EXPECT_EQ(x.rpart(), 2);
    EXPECT_EQ(x.dpart(), 0);
  }
  {
    std::stringstream s("(2_e) ");
    duald x(1,8);
    s >> x;
    EXPECT_TRUE(s);
    EXPECT_EQ(x.rpart(), 0);
    EXPECT_EQ(x.dpart(), 2);
  }
  {
    std::stringstream s("(3_e ) ");
    duald x(1,8);
    s >> x;
    EXPECT_TRUE(s);
    EXPECT_EQ(x.rpart(), 0);
    EXPECT_EQ(x.dpart(), 3);
  }
  {
    std::stringstream s(" (2.3+ 3_el)");
    dualld x;
    s >> x;
    EXPECT_TRUE(s);
    EXPECT_NEAR(0, (x - (2.3 + 3_el)).rpart(), 10000*std::numeric_limits<long double>::epsilon());
  }
#ifndef _MSC_VER // hoping for a patch to fix this
  {
    std::stringstream s("((2+3_ef),(4+5_ef))");
    cdualf x;
    s >> x;
    EXPECT_TRUE(s);
    EXPECT_EQ(x, cdualf(2 + 3_ef, 4 + 5_ef));
    EXPECT_EQ(x.real().rpart(), 2);
    EXPECT_EQ(x.real().dpart(), 3);
    EXPECT_EQ(x.imag().rpart(), 4);
    EXPECT_EQ(x.imag().dpart(), 5);
  }
#endif
  {
    std::stringstream s("1 ");
    cdualf x;
    s >> x;
    EXPECT_TRUE(s);
    EXPECT_EQ(x.real().rpart(), 1);
    EXPECT_EQ(x.real().dpart(), 0);
    EXPECT_EQ(x.imag().rpart(), 0);
    EXPECT_EQ(x.imag().dpart(), 0);
  }

  // invalid conversions
  {
    std::stringstream s("(2 -3.3_e ) ");
    duald x;
    s.setstate(std::ios_base::failbit);
    s >> x;
    EXPECT_FALSE(s);
    EXPECT_NE(x.rpart(), 2);
    EXPECT_NE(x.dpart(), -3.3);
  }
  {
    std::stringstream s("");
    duald x;
    s >> x;
    EXPECT_FALSE(s);
    EXPECT_DEQ(x, duald());
  }
  {
    std::stringstream s("()");
    duald x;
    s >> x;
    EXPECT_FALSE(s);
    EXPECT_DNE(x, 2+3.3_e);
  }
  {
    std::stringstream s("(a)");
    duald x;
    s >> x;
    EXPECT_FALSE(s);
    EXPECT_DNE(x, 2+3.3_e);
  }
  {
    std::stringstream s("asdf");
    duald x;
    s >> x;
    EXPECT_FALSE(s);
    EXPECT_DNE(x, 2+3.3_e);
  }
  {
    std::stringstream s("(2+ _ef)");
    duald x;
    s >> x;
    EXPECT_FALSE(s);
    EXPECT_DNE(x, 2+3.3_e);
  }
  {
    std::stringstream s("(2+ 3_)");
    duald x;
    s >> x;
    EXPECT_FALSE(s);
    EXPECT_DNE(x, 2+3_e);
  }
  {
    std::stringstream s("(2+ 3.3_ef");
    dualf x;
    s >> x;
    EXPECT_FALSE(s);
    EXPECT_DNE(x, 2+3.3_ef);
  }
  {
    std::stringstream s("(2+ 3.3_ef}");
    dualf x;
    s >> x;
    EXPECT_FALSE(s);
    EXPECT_DNE(x, 2+3.3_ef);
  }
  {
    std::stringstream s("(2+ 3.3_el)");
    duald x;
    s >> x;
    EXPECT_FALSE(s);
    EXPECT_DNE(x, 2+3.3_e);
  }
  {
    std::stringstream s("(2+ 3.3_el)");
    dualf x;
    s >> x;
    EXPECT_FALSE(s);
    EXPECT_DNE(x, 2+3.3_ef);
  }
  {
    std::stringstream s("(2+ 3.3_ef)");
    dualld x;
    s >> x;
    EXPECT_FALSE(s);
    EXPECT_DNE(x, 2+3.3_el);
  }
  {
    std::stringstream s("(2,3.3_ef)");
    dualld x;
    s >> x;
    EXPECT_FALSE(s);
    EXPECT_DNE(x, 2+3.3_e);
  }
  {
    std::stringstream s("(2+3.3_f)");
    dualld x;
    s >> x;
    EXPECT_FALSE(s);
    EXPECT_DNE(x, 2+3.3_el);
  }
  {
    std::stringstream s("(2+3.3)");
    dualld x;
    s >> x;
    EXPECT_FALSE(s);
    EXPECT_DNE(x, 2+3.3_el);
  }
  {
    std::stringstream s("(2-3.3)");
    dualld x;
    s >> x;
    EXPECT_FALSE(s);
    EXPECT_DNE(x, 2-3.3_el);
  }
  {
    std::stringstream s("(1 ");
    cdualf x;
    s >> x;
    EXPECT_FALSE(s);
    EXPECT_EQ(x, cdualf());
  }
  {
    std::stringstream s("(1_) ");
    cdualf x;
    s >> x;
    EXPECT_FALSE(s);
    EXPECT_EQ(x, cdualf());
  }
  {
    std::stringstream s("(1.3 _e) ");
    cduald x;
    s >> x;
    EXPECT_FALSE(s);
    EXPECT_EQ(x, cduald());
  }

}

TEST(non_class, math) {
  dualf x = 2 + 4_e;
  hyperdualf y = 3 + 5_ef;
  // TODO: FIXME: check results too
  y = x + 1;
  y = 1 + x;
  y = x + x;
  y = y + x;
  y = y + 1;
  y = 1 + y;

  y = x - 1;
  y = 1 - x;
  y = x - x;
  y = y - x;
  y = y - 1;
  y = 1 - y;

  y = x * 1;
  y = 1 * x;
  y = x * x;
  y = y * x;
  y = y * 1;
  y = 1 * y;

  y = x / 1;
  y = 1 / x;
  y = x / x;
  y = y / x;
  y = y / 1;
  y = 1 / y;
}

TEST(non_class, random) {
  typedef dualf Rt;
  Rt a = duals::random<Rt>(0.1, 0.99+0.88_ef);
  EXPECT_NE(a.rpart(), 0);
  EXPECT_NE(a.dpart(), 0);

  a = duals::random<Rt>(0.1, 0.99);
  EXPECT_NE(a.rpart(), 0);
  EXPECT_EQ(a.dpart(), 0);

  Rt b = duals::random<Rt, std::cauchy_distribution<float> >(0.1, 0.99+0.2_ef);
  EXPECT_NE(b.rpart(), 0);
  EXPECT_NE(b.dpart(), 0);

  Rt c = duals::random<Rt>();
  EXPECT_NE(c.rpart(), 0);
  EXPECT_EQ(c.dpart(), 0);
}

TEST(non_class, random2) {
  // float
  float a1 = duals::randos::random2<float>();
  float a2 = duals::randos::random2<float>();
  EXPECT_NE(a1, 0);
  EXPECT_NE(a2, 0);
  EXPECT_NE(a1, a2);

  // complexf
  complexf b1 = duals::randos::random2<complexf>();
  complexf b2 = duals::randos::random2<complexf>();
  EXPECT_NE(b1.real(), 0);
  EXPECT_NE(b1.imag(), 0);
  EXPECT_NE(b2.real(), 0);
  EXPECT_NE(b2.imag(), 0);
  EXPECT_NE(b1.real(), b2.real());
  EXPECT_NE(b1.imag(), b2.imag());

  // dualf
  dualf c1 = duals::randos::random2<dualf>();
  dualf c2 = duals::randos::random2<dualf>();
  EXPECT_NE(c1.rpart(), 0);
  EXPECT_NE(c1.dpart(), 0);
  EXPECT_NE(c2.rpart(), 0);
  EXPECT_NE(c2.dpart(), 0);
  EXPECT_NE(c1.rpart(), c2.rpart());
  EXPECT_NE(c1.dpart(), c2.dpart());

  // cdualf
  cdualf d1 = duals::randos::random2<cdualf>();
  cdualf d2 = duals::randos::random2<cdualf>();
  EXPECT_NE(d1.real().rpart(), 0);
  EXPECT_NE(d1.real().dpart(), 0);
  EXPECT_NE(d1.imag().rpart(), 0);
  EXPECT_NE(d1.imag().dpart(), 0);

  EXPECT_NE(d2.real().rpart(), 0);
  EXPECT_NE(d2.real().dpart(), 0);
  EXPECT_NE(d2.imag().rpart(), 0);
  EXPECT_NE(d2.imag().dpart(), 0);

  EXPECT_NE(d1.real().rpart(), d2.real().rpart());
  EXPECT_NE(d1.real().dpart(), d2.real().dpart());

  EXPECT_NE(d1.imag().rpart(), d2.imag().rpart());
  EXPECT_NE(d1.imag().dpart(), d2.imag().dpart());
}

TEST(smoke, funcs) {
  dualf x = pow(1 + 2_e, 2);
  dualf y(2,3);
  x = pow(x,y);
  x = pow(2,y);
  x = pow(y,2);
  x = pow(2 + 2_e, -1);
  EXPECT_EQ(x.rpart(), 0.5);

  x = 2 + 4_e;
  hyperdualf X = pow(x,y);
  auto Y = pow(X,2);
  EXPECT_EQ(Y.rpart().rpart(), 16);

  // TODO / FIXME : check results of these
  // TODO / FIXME : check for hyperdual and cdual args too
  x = exp(y);
  x = log(y);
  x = log10(y);
  x = pow(x, y);
  x = pow(x, 2);
  x = pow(2, x);
  X = pow(x, Y);
  X = pow(Y, 2);
  X = pow(2, Y);
  x = abs(y);
  x = sqrt(y);
  x = cbrt(y);
  x = sin(y);
  x = cos(y);
  x = tan(y);
  x = asin(y);
  x = acos(y);
  x = atan(y);

  x = exp(log(log10(pow(x,
                        abs(sqrt(sin(cos(tan(asin(acos(atan(y))))))))))));

  // TODO:
  //x = sinh(y);
  //x = cosh(y);
  //x = tanh(y);
  //x = asinh(y);
  //x = acosh(y);
  //x = atanh(y);

}

TEST(complex, nesting) {
  complexf a;
  complexf b;
  complexf c;
  std::complex<dualf> A;
  std::complex<dualf> B(2);
  std::complex<dualf> C(3+4_e, 5+6_ef);

  A = B + C;
  A = B - C;
  A = B * C;
  A = B / C;

  a = duals::rpart(C);
  EXPECT_EQ(a, complexf(3,5));
  a = duals::dpart(C);
  EXPECT_EQ(a, complexf(4,6));

  c = duals::rpart(a);
  EXPECT_EQ(a, c);
  c = duals::dpart(a);
  EXPECT_EQ(c, complexf(0,0));

  cdualf z(2+4_ef,3+6_ef);
  EXPECT_EQ(duals::rpart(z), complexf(2,3));
  EXPECT_EQ(duals::dpart(z), complexf(4,6));

  EXPECT_EQ(duals::rpart(9), 9);
  EXPECT_EQ(duals::dpart(9), 0);
}

TEST(complex, mixing) {
  complexf a;
  complexf b;
  complexf c;
  std::complex<dualf> A;
  std::complex<dualf> B(2);
  std::complex<dualf> C(3+4_e, 5+6_ef);

  A = 1;
  A *= 2;
  A *= 2.f;
  A *= 2_ef;

  //A = a + 1_ef;

  // complex<dual> * dual -> complex<dual>
  B = A * a;
  // complex<dual> * real -> complex<dual>
  // complex<real> * dual -> complex<dual> ?

}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  std::cout.precision(20);
  std::cerr.precision(20);
  return RUN_ALL_TESTS();
}

//===-- test_packets.cpp - test duals/dual_eigen -----------------*- C++ -*-===//
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
 * \file test_packets Dual number Eigen integration tests
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
#define EXPECT_DEQ(A,B) EXPECT_EQ(rpart(A), rpart(B)); EXPECT_EQ(dpart(A), dpart(B))
#define ASSERT_DEQ(A,B) ASSERT_EQ(rpart(A), rpart(B)); ASSERT_EQ(dpart(A), dpart(B))
#define EXPECT_DNE(A,B) EXPECT_NE(rpart(A), rpart(B)); EXPECT_NE(dpart(A), dpart(B))
#define EXPECT_DNEAR(A,B,tol)                      \
  EXPECT_NEAR(abs(rpart((A) - (B))),0,tol);        \
  EXPECT_NEAR(abs(dpart((A) - (B))),0,tol)


#if !defined(CPPDUALS_DONT_VECTORIZE) && !defined(EIGEN_DONT_VECTORIZE)

TEST(Packet1cdf, pload_pstore) {
  using namespace Eigen::internal;
  cdualf cd1 = cdualf(1+2_ef,3+4_ef);
  cdualf cd2 = cdualf(5+6_ef,7+8_ef);
  Packet1cdf p1 = pload<Packet1cdf>(&cd1);
  pstore(&cd2, p1);
  EXPECT_DEQ(cd1, cd2);
}

using duals::randos::random2;

#define GEN_PACKET_TEST_BI(PTYPE,pop,op) TEST(PTYPE,pop) {              \
    using namespace Eigen::internal;                                    \
    typedef unpacket_traits<PTYPE>::type DTYPE;                         \
    const static int N = unpacket_traits<PTYPE>::size;                  \
    double tol = rpart(20*Eigen::NumTraits<DTYPE>::epsilon());          \
    std::vector<DTYPE,Eigen::aligned_allocator<DTYPE>> cd1(N);          \
    std::vector<DTYPE,Eigen::aligned_allocator<DTYPE>> cd2(N);          \
    std::vector<DTYPE,Eigen::aligned_allocator<DTYPE>> cd3(N);          \
    for (size_t i = 0; i < N; i++) {                                    \
      cd1[i] = random2<DTYPE>();                                        \
      cd2[i] = random2<DTYPE>();                                        \
    }                                                                   \
    PTYPE p1 = pload<PTYPE>(cd1.data());                                \
    PTYPE p2 = pload<PTYPE>(cd2.data());                                \
    auto p3 = pop (p1, p2);                                             \
    pstore(cd3.data(), p3);                                             \
    for (int i = 0; i < N; i++) {                                       \
      EXPECT_DNEAR(cd3[i], cd1[i] op cd2[i], tol)                       \
        << cd1[i] << ',' << cd2[i] << " fail at " << i;                 \
    }                                                                   \
  }

#define GEN_PACKET_TEST_UN(PTYPE,pop,op) TEST(PTYPE,pop) {      \
    using namespace Eigen::internal;                                    \
    typedef unpacket_traits<PTYPE>::type DTYPE;                         \
    const static int N = unpacket_traits<PTYPE>::size;                  \
    double tol = rpart(20*Eigen::NumTraits<DTYPE>::epsilon());          \
    std::vector<DTYPE,Eigen::aligned_allocator<DTYPE>> cd1(N);          \
    std::vector<DTYPE,Eigen::aligned_allocator<DTYPE>> cd2(N);          \
    for (size_t i = 0; i < N; i++) {                                    \
      cd1[i] = random2<DTYPE>();                                        \
    }                                                                   \
    PTYPE p1 = pload<PTYPE>(cd1.data());                                \
    auto p2 = pop (p1);                                                 \
    pstore(cd2.data(), p2);                                             \
    for (int i = 0; i < N; i++) {                                       \
      EXPECT_DNEAR(cd2[i], op(cd1[i]), tol)                             \
        << cd1[i] << " fail at " << i;                                  \
    }                                                                   \
  }

#define GEN_PACKET_TEST_RD(PTYPE,pop,op) TEST(PTYPE,pop) {      \
    using namespace Eigen::internal;                                    \
    typedef unpacket_traits<PTYPE>::type DTYPE;                         \
    const static int N = unpacket_traits<PTYPE>::size;                  \
    double tol = rpart(20*Eigen::NumTraits<DTYPE>::epsilon());          \
    std::vector<DTYPE,Eigen::aligned_allocator<DTYPE>> cd1(N);          \
    std::vector<DTYPE,Eigen::aligned_allocator<DTYPE>> cd2(N);          \
    for (size_t i = 0; i < N; i++) {                                    \
      cd1[i] = random2<DTYPE>();                                        \
    }                                                                   \
    PTYPE p1 = pload<PTYPE>(cd1.data());                                \
    DTYPE p2 = pop (p1);                                                \
    DTYPE acc(1);                                                       \
    for (int i = 0; i < N; i++)                                         \
      acc op##= cd1[i];                                                 \
    EXPECT_DNEAR(acc, (DTYPE(1) op p2), tol)                            \
      << acc << " " << p2 << " fail.";                                  \
  }

#define GEN_PACKET_TEST_REVERSE(PTYPE) TEST(PTYPE,preverse) {   \
    using namespace Eigen::internal;                                    \
    typedef unpacket_traits<PTYPE>::type DTYPE;                         \
    const static int N = unpacket_traits<PTYPE>::size;                  \
    std::vector<DTYPE,Eigen::aligned_allocator<DTYPE>> cd1(N);          \
    std::vector<DTYPE,Eigen::aligned_allocator<DTYPE>> cd2(N);          \
    for (size_t i = 0; i < N; i++) {                                    \
      cd1[i] = random2<DTYPE>();                                        \
    }                                                                   \
    PTYPE p1 = pload<PTYPE>(cd1.data());                                \
    PTYPE p2 = preverse (p1);                                           \
    pstore(cd2.data(), p2);                                             \
    for (int i = 0; i < N; i++) {                                       \
      EXPECT_DEQ(cd1[i], cd2[N-1-i])                                    \
        << cd1[i] << "," << cd2[N-1-i] << " @ " << i << " fail.";       \
    }                                                                   \
  }

#define GEN_PACKET_TEST_FIRST(PTYPE) TEST(PTYPE,pfirst) {       \
    using namespace Eigen::internal;                                    \
    typedef unpacket_traits<PTYPE>::type DTYPE;                         \
    const static int N = unpacket_traits<PTYPE>::size;                  \
    std::vector<DTYPE,Eigen::aligned_allocator<DTYPE>> cd1(N);          \
    for (size_t i = 0; i < N; i++) {                                    \
      cd1[i] = random2<DTYPE>();                                        \
    }                                                                   \
    PTYPE p1 = pload<PTYPE>(cd1.data());                                \
    DTYPE c2 = pfirst (p1);                                             \
    EXPECT_DEQ(cd1[0], c2)                                              \
      << cd1[0] << "," << c2 << " fail.";                               \
  }

#define GEN_PACKET_TEST_SET1(PTYPE) TEST(PTYPE,pset1) {                 \
    using namespace Eigen::internal;                                    \
    typedef unpacket_traits<PTYPE>::type DTYPE;                         \
    const static int N = unpacket_traits<PTYPE>::size;                  \
    std::vector<DTYPE,Eigen::aligned_allocator<DTYPE>> cd1(N);          \
    std::vector<DTYPE,Eigen::aligned_allocator<DTYPE>> cd2(N);          \
    for (size_t i = 0; i < N; i++) {                                    \
      cd1[i] = random2<DTYPE>();                                        \
      cd2[i] = random2<DTYPE>();                                        \
    }                                                                   \
    DTYPE c2 = random2<DTYPE>();                                        \
    PTYPE p1 = pset1<PTYPE>(c2);                                        \
    PTYPE p2 = pload1<PTYPE>(&c2);                                      \
    pstore(cd1.data(), p1);                                             \
    pstore(cd2.data(), p2);                                             \
    for (int i = 0; i < N; i++) {                                       \
      EXPECT_DEQ(cd1[i], c2)                                            \
        << i << ":" << cd1[i] << "," << c2 << " fail.";                 \
      EXPECT_DEQ(cd2[i], c2)                                            \
        << i << ":" << cd1[i] << "," << c2 << " fail.";                 \
    }                                                                   \
  }

#define GEN_PACKET_TEST_LOADDUP(PTYPE) TEST(PTYPE,ploaddup) {           \
    using namespace Eigen::internal;                                    \
    typedef unpacket_traits<PTYPE>::type DTYPE;                         \
    const static int N = unpacket_traits<PTYPE>::size;                  \
    std::vector<DTYPE,Eigen::aligned_allocator<DTYPE>> cd1(1+N/2);      \
    std::vector<DTYPE,Eigen::aligned_allocator<DTYPE>> cd2(N);          \
    for (size_t i = 0; i < 1+N/2; i++) { cd1[i] = random2<DTYPE>(); }   \
    for (size_t i = 0; i < N    ; i++) { cd2[i] = random2<DTYPE>(); }   \
    PTYPE p1 = ploaddup<PTYPE>(cd1.data());                             \
    pstore(cd2.data(), p1);                                             \
    for (int i = 0; i < N; i++) {                                       \
      EXPECT_DEQ(cd1[i/2], cd2[i])                                      \
        << i << ":" << cd1[i/2] << "," << cd2[i] << " fail.";           \
    }                                                                   \
  }

#define GEN_PACKET_TEST_ULOAD(PTYPE) TEST(PTYPE,ploadu) {               \
    using namespace Eigen::internal;                                    \
    typedef unpacket_traits<PTYPE>::type DTYPE;                         \
    const static int N = unpacket_traits<PTYPE>::size;                  \
    char b1[sizeof(DTYPE) * (N+2)];                                     \
    char b2[sizeof(DTYPE) * (N+2)];                                     \
    DTYPE * cd1 = (DTYPE *)&b1[1];                                      \
    DTYPE * cd2 = (DTYPE *)&b2[2];                                      \
    for (size_t i = 0; i < N+1; i++) {                                  \
      cd1[i] = random2<DTYPE>();                                        \
      cd2[i] = random2<DTYPE>();                                        \
    }                                                                   \
    PTYPE p1 = ploadu<PTYPE>(&cd1[1]);                                  \
    pstoreu(&cd2[1], p1);                                               \
    for (int i = 1; i < N+1; i++) {                                     \
      EXPECT_DEQ(cd1[i], cd2[i])                                        \
        << cd1[i] << "," << cd2[i] << " fail.";                         \
    }                                                                   \
  }

#define GEN_PACKET_TEST_BROADCAST(PTYPE,B) TEST(PTYPE,pbroadcast##B) {  \
    using namespace Eigen::internal;                                    \
    typedef unpacket_traits<PTYPE>::type DTYPE;                         \
    const static int N = unpacket_traits<PTYPE>::size;                  \
    PTYPE p1;                                                           \
    std::vector<DTYPE,Eigen::aligned_allocator<DTYPE>> cd1(B);          \
    std::vector<DTYPE,Eigen::aligned_allocator<DTYPE>> cd2(N);          \
    for (size_t i = 0; i < N; i++) { cd1[i] = random2<DTYPE>(); }       \
    PTYPE p0,p3,p2;                                                     \
    if (B == 2) pbroadcast2(cd1.data(), p0,p1);                         \
    if (B == 4) pbroadcast4(cd1.data(), p0,p1,p2,p3);                   \
    pstore(cd2.data(), p0);                                             \
    for (int i = 0; i < N; i++) {                                       \
      EXPECT_DEQ(cd1[0], cd2[i]) << i << ":" << cd1[0] << "," << cd2[i] << " fail."; \
    }                                                                   \
    pstore(cd2.data(), p1);                                             \
    for (int i = 0; i < N; i++) {                                       \
      EXPECT_DEQ(cd1[1], cd2[i]) << i << ":" << cd1[1] << "," << cd2[i] << " fail."; \
    }                                                                   \
    if (B == 4) {                                                       \
      pstore(cd2.data(), p2);                                           \
      for (int i = 0; i < N; i++) {                                     \
        EXPECT_DEQ(cd1[2], cd2[i]) << i << ":" << cd1[2] << "," << cd2[i] << " fail."; \
      }                                                                 \
      pstore(cd2.data(), p3);                                           \
      for (int i = 0; i < N; i++) {                                     \
        EXPECT_DEQ(cd1[3], cd2[i]) << i << ":" << cd1[3] << "," << cd2[i] << " fail."; \
      }                                                                 \
    }                                                                   \
  }

#define GEN_PACKET_TEST_CPLXFLIP(PTYPE) TEST(PTYPE,pcplxflip) {         \
    using namespace Eigen::internal;                                    \
    typedef unpacket_traits<PTYPE>::type DTYPE;                         \
    const static int N = unpacket_traits<PTYPE>::size;                  \
    std::vector<DTYPE,Eigen::aligned_allocator<DTYPE>> cd1(N);          \
    std::vector<DTYPE,Eigen::aligned_allocator<DTYPE>> cd2(N);          \
    for (size_t i = 0; i < N; i++) {                                    \
      cd1[i] = random2<DTYPE>();                                        \
      cd2[i] = random2<DTYPE>();                                        \
    }                                                                   \
    PTYPE p1 = pload<PTYPE>(cd1.data());                                \
    PTYPE p2 = pcplxflip (p1);                                          \
    pstore(cd2.data(), p2);                                             \
    for (int i = 0; i < N; i++) {                                       \
      EXPECT_DEQ(real(cd1[i]), imag(cd2[i]))                            \
        << cd1[i] << "," << cd2[i] << " fail.";                         \
      EXPECT_DEQ(imag(cd1[i]), real(cd2[i]))                            \
        << cd1[i] << "," << cd2[i] << " fail.";                         \
    }                                                                   \
  }

#define GEN_PACKET_TEST_CONJ_HELPER(PTYPE) TEST(PTYPE,conj_helper) {    \
    using namespace Eigen::internal;                                    \
    typedef unpacket_traits<PTYPE>::type DTYPE;                         \
    const static int N = unpacket_traits<PTYPE>::size;                  \
    double tol = rpart(10*Eigen::NumTraits<DTYPE>::epsilon());          \
    std::vector<DTYPE,Eigen::aligned_allocator<DTYPE>> cd1(N);          \
    std::vector<DTYPE,Eigen::aligned_allocator<DTYPE>> cd2(N);          \
    std::vector<DTYPE,Eigen::aligned_allocator<DTYPE>> cd3(N);          \
    std::vector<DTYPE,Eigen::aligned_allocator<DTYPE>> r(N);            \
    for (size_t i = 0; i < N; i++) {                                    \
      cd1[i] = random2<DTYPE>();                                        \
      cd2[i] = random2<DTYPE>();                                        \
      cd3[i] = random2<DTYPE>();                                        \
    }                                                                   \
    PTYPE p1 = pload<PTYPE>(cd1.data());                                \
    PTYPE p2 = pload<PTYPE>(cd2.data());                                \
    PTYPE p3 = pload<PTYPE>(cd3.data());                                \
    conj_helper<PTYPE,PTYPE,false,false> cj; p3 = cj.pmadd(p1,p2,p3);   \
    pstore(r.data(), p3);                                               \
    for (int i = 0; i < N; i++) {                                       \
      EXPECT_DNEAR(r[i], cd1[i] * cd2[i] + cd3[i], tol)                 \
        << r[i] << "!=" << cd1[i] << "*" << cd3[i] << "+" << cd3[i] << " fail."; \
    }                                                                   \
  }


#define GEN_PACKET_TESTS(PTYPE)          \
  GEN_PACKET_TEST_BI(PTYPE,padd,+)       \
  GEN_PACKET_TEST_BI(PTYPE,psub,-)       \
  GEN_PACKET_TEST_BI(PTYPE,pmul,*)       \
  GEN_PACKET_TEST_BI(PTYPE,pdiv,/)       \
  GEN_PACKET_TEST_UN(PTYPE,pnegate,-)    \
  GEN_PACKET_TEST_RD(PTYPE,predux,+)     \
  GEN_PACKET_TEST_RD(PTYPE,predux_mul,*) \
  GEN_PACKET_TEST_ULOAD(PTYPE)           \
  GEN_PACKET_TEST_SET1(PTYPE)            \
  GEN_PACKET_TEST_LOADDUP(PTYPE)         \
  /*GEN_PACKET_TEST_BROADCAST(PTYPE,2)*/ \
  GEN_PACKET_TEST_BROADCAST(PTYPE,4)     \
  GEN_PACKET_TEST_FIRST(PTYPE)           \
  GEN_PACKET_TEST_REVERSE(PTYPE)

#define GEN_CPACKET_TESTS(PTYPE)         \
  GEN_PACKET_TESTS(PTYPE)                \
  GEN_PACKET_TEST_CPLXFLIP(PTYPE)        \
  GEN_PACKET_TEST_CONJ_HELPER(PTYPE)     \
  GEN_PACKET_TEST_UN(PTYPE,pconj,conj)

// TODO:
//pcplxflip
//preduxp
//pand
//por
//pxor
//andnot
//pbroadcast4
//ploadquad (for packets w/ size==8)
//pgather
//pscatter
//align_impl
//insertfirst
//insertlast


// test the tests
GEN_PACKET_TESTS(Packet4f)
GEN_CPACKET_TESTS(Packet2cf)
#if defined(EIGEN_VECTORIZE_AVX)
GEN_CPACKET_TESTS(Packet4cf)
#endif

#ifdef EIGEN_VECTORIZE_SSE
GEN_PACKET_TESTS(Packet2df)
GEN_PACKET_TESTS(Packet1dd)
GEN_CPACKET_TESTS(Packet1cdf)
#endif

#if defined(EIGEN_VECTORIZE_AVX)
GEN_PACKET_TESTS(Packet4df)
GEN_PACKET_TESTS(Packet2dd)
GEN_CPACKET_TESTS(Packet2cdf)
#if defined(__AVX2__)
GEN_CPACKET_TESTS(Packet1cdd)
#endif
#endif

#endif // CPPDUALS_DONT_VECTORIZE

TEST(compile, VECTORIZE) {
#if defined(CPPDUALS_DONT_VECTORIZE) || defined(EIGEN_DONT_VECTORIZE)
  EXPECT_TRUE(false) << "CPPDUALS_DONT_VECTORIZE supresses vectorization tests!";
#endif
}
TEST(compile, SSE) {
  EXPECT_TRUE(std::string(Eigen::SimdInstructionSetsInUse()).find("SSE") != std::string::npos)
    << "Not using SSE instructions:" << Eigen::SimdInstructionSetsInUse();
#ifndef EIGEN_VECTORIZE_SSE
  EXPECT_TRUE(false)
    << "Not using EIGEN_VECTORIZE_SSE:" << Eigen::SimdInstructionSetsInUse();
#endif
}
TEST(compile, AVX) {
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

// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2019 Michael Tesch <tesch1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CDUAL_AVX_H
#define EIGEN_CDUAL_AVX_H

namespace Eigen {

namespace internal {

#define vec8f_swizzle1(v,p,q,r,s)                     \
  (_mm256_permute_ps(v, ((s)<<6|(r)<<4|(q)<<2|(p))))

#ifdef __AVX2__
#define vec4d_swizzle1(v,p,q,r,s)                     \
  (_mm256_permute4x64_pd(v,(s)<<6|(r)<<4|(q)<<2|(p)))
#else
//#error "TODO"
#endif


//---------- float ----------
struct Packet2cdf
{
  EIGEN_STRONG_INLINE Packet2cdf() {}
  EIGEN_STRONG_INLINE explicit Packet2cdf(const __m256 & a) : v(a) {}
  EIGEN_STRONG_INLINE explicit Packet2cdf(const Packet4df & a) : v(a.v) {}
  //__m256  v;
  Packet4df v;
};

// Use the packet_traits defined in AVX/PacketMath.h instead if we're going
// to leverage AVX instructions.
#if !defined(CPPDUALS_DONT_VECTORIZE_CDUAL)
template<> struct packet_traits<std::complex<duals::dual<float> > >  : default_packet_traits
{
  typedef Packet2cdf type;
  typedef Packet1cdf half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 0,
    size = 2,
    HasHalfPacket = 1,

    HasAdd    = 1,
    HasSub    = 1,
    HasMul    = 1,
    HasDiv    = 1,
    HasNegate = 1,
    HasAbs    = 0,
    HasAbs2   = 0,
    HasMin    = 0,
    HasMax    = 0,
    HasSetLinear = 0
  };
};
#endif

template<> struct unpacket_traits<Packet2cdf> {
  typedef std::complex<duals::dual<float> > type;
  enum {size=2, alignment=Aligned32, masked_load_available=false, masked_store_available=false, vectorizable=true};
  typedef Packet1cdf half;
};

template<> EIGEN_STRONG_INLINE Packet2cdf padd<Packet2cdf>(const Packet2cdf& a, const Packet2cdf& b)
{ return Packet2cdf(_mm256_add_ps(a.v.v,b.v.v)); }
template<> EIGEN_STRONG_INLINE Packet2cdf psub<Packet2cdf>(const Packet2cdf& a, const Packet2cdf& b)
{ return Packet2cdf(_mm256_sub_ps(a.v.v,b.v.v)); }
template<> EIGEN_STRONG_INLINE Packet2cdf pnegate(const Packet2cdf& a) { return Packet2cdf(pnegate(a.v.v)); }
template<> EIGEN_STRONG_INLINE Packet2cdf pconj(const Packet2cdf& a)
{
  const __m256 mask = _mm256_castsi256_ps(_mm256_setr_epi32(0x00000000,0x00000000,0x80000000,0x80000000,
                                                            0x00000000,0x00000000,0x80000000,0x80000000));
  return Packet2cdf(_mm256_xor_ps(a.v.v,mask));
}

template<> EIGEN_STRONG_INLINE std::complex<duals::dual<float> >  pfirst<Packet2cdf>(const Packet2cdf& a)
{
  EIGEN_ALIGN32 float res[8];
  _mm256_store_ps(res, a.v.v);
  return std::complex<duals::dual<float> >(duals::dual<float>(res[0],res[1]),
                                           duals::dual<float>(res[2],res[3]));
}

template<> EIGEN_STRONG_INLINE Packet2cdf pmul<Packet2cdf>(const Packet2cdf& a, const Packet2cdf& b)
{
#ifdef __FMA__xx
  const __m256 mask = _mm256_castsi256_ps(_mm256_setr_epi32(0x00000000,0x00000000,0xffffffff,0xffffffff,
                                                            0x00000000,0x00000000,0xffffffff,0xffffffff));
  Packet2cdf x(vec8f_swizzle1
               (_mm256_addsub_ps(_mm256_fmadd_ps(vec8f_swizzle1(a.v.v, 0, 0, 0, 0),
                                                 vec8f_swizzle1(b.v.v, 0, 2, 1, 3),
                                                 _mm256_and_ps(mask,
                                                               _mm256_mul_ps(vec8f_swizzle1(a.v.v, 0, 0, 1, 1),
                                                                             vec8f_swizzle1(b.v.v, 0, 0, 0, 2)))),
                                 _mm256_fmadd_ps(vec8f_swizzle1(a.v.v, 2, 2, 2, 2),
                                                 vec8f_swizzle1(b.v.v, 2, 0, 3, 1),
                                                 _mm256_and_ps(mask,
                                                               _mm256_mul_ps(vec8f_swizzle1(a.v.v, 0, 0, 3, 3),
                                                                             vec8f_swizzle1(b.v.v, 0, 0, 2, 0))))),
                0,2,1,3));
  return x;
#else
  // help gcc
  __m256 y0 = a.v.v;
  __m256 y1 = _mm256_permute_ps(y0,0);
  __m256 y2 = b.v.v;
  __m256 y3 = _mm256_permute_ps(y2,216);
  y1 = _mm256_mul_ps(y1,y3);
  y3 = _mm256_permute_ps(y0,84);
  __m256 y4 = _mm256_permute_ps(y2,132);
  y3 = _mm256_mul_ps(y3,y4);
  y4 = _mm256_setzero_ps();
  y3 = _mm256_blend_ps(y4,y3,204);
  y1 = _mm256_add_ps(y1,y3);
  y3 = _mm256_permute_ps(y0,170);
  __m256 y5 = _mm256_permute_ps(y2,114);
  y3 = _mm256_mul_ps(y3,y5);
  y0 = _mm256_movehdup_ps(y0);
  y2 = _mm256_permute_ps(y2,36);
  y0 = _mm256_mul_ps(y0,y2);
  y0 = _mm256_blend_ps(y4,y0,204);
  y0 = _mm256_add_ps(y3,y0);
  y0 = _mm256_addsub_ps(y1,y0);
  y0 = _mm256_permute_ps(y0,216);
  return Packet2cdf(y0);

#endif
}

template<> EIGEN_STRONG_INLINE Packet2cdf pand   <Packet2cdf>(const Packet2cdf& a, const Packet2cdf& b)
{ return Packet2cdf(_mm256_and_ps(a.v.v,b.v.v)); }
template<> EIGEN_STRONG_INLINE Packet2cdf por    <Packet2cdf>(const Packet2cdf& a, const Packet2cdf& b)
{ return Packet2cdf(_mm256_or_ps(a.v.v,b.v.v)); }
template<> EIGEN_STRONG_INLINE Packet2cdf pxor   <Packet2cdf>(const Packet2cdf& a, const Packet2cdf& b)
{ return Packet2cdf(_mm256_xor_ps(a.v.v,b.v.v)); }
template<> EIGEN_STRONG_INLINE Packet2cdf pandnot<Packet2cdf>(const Packet2cdf& a, const Packet2cdf& b)
{ return Packet2cdf(_mm256_andnot_ps(a.v.v,b.v.v)); }

template<> EIGEN_STRONG_INLINE Packet2cdf pload <Packet2cdf>(const std::complex<duals::dual<float> >* from)
{ EIGEN_DEBUG_ALIGNED_LOAD return Packet2cdf(pload<Packet8f>((const float*)from)); }
template<> EIGEN_STRONG_INLINE Packet2cdf ploadu<Packet2cdf>(const std::complex<duals::dual<float> >* from)
{
  EIGEN_DEBUG_UNALIGNED_LOAD return Packet2cdf(_mm256_loadu_ps((float*)from));
}
template<> EIGEN_STRONG_INLINE Packet2cdf pset1<Packet2cdf>(const std::complex<duals::dual<float> >&  from)
{
  /* here we really have to use unaligned loads :( */
  const __m128 v = ploadu<Packet4f>((float *)&from);
  return Packet2cdf(_mm256_insertf128_ps(_mm256_castps128_ps256(v),v,1));
  //return Packet2cdf(_mm256_set_m128(v,v)); // missing on older GCCs
}

template<> EIGEN_STRONG_INLINE Packet2cdf ploaddup<Packet2cdf>(const std::complex<duals::dual<float> >* from)
{ return pset1<Packet2cdf>(*from); }

// FIXME force unaligned store, this is a temporary fix
template<> EIGEN_STRONG_INLINE void
pstore <std::complex<duals::dual<float> > >(std::complex<duals::dual<float> > *   to, const Packet2cdf& from)
{ EIGEN_DEBUG_ALIGNED_STORE pstore((float*)to, Packet8f(from.v.v)); }
template<> EIGEN_STRONG_INLINE void
pstoreu<std::complex<duals::dual<float> > >(std::complex<duals::dual<float> > *   to, const Packet2cdf& from)
{ EIGEN_DEBUG_UNALIGNED_STORE pstoreu((float*)to, Packet8f(from.v.v)); }

//template<> EIGEN_STRONG_INLINE void
//prefetch<std::complex<duals::dual<float> > >(const std::complex<duals::dual<float> > *   addr)
//{ _mm256_prefetch((SsePrefetchPtrType)(addr), _MM_HINT_T0); }

template<> EIGEN_STRONG_INLINE Packet2cdf preverse(const Packet2cdf& a)
{
  const __m256 result = _mm256_permute2f128_ps(a.v.v, a.v.v, 1);
  return Packet2cdf(result);
}

template<> EIGEN_STRONG_INLINE std::complex<duals::dual<float> > predux<Packet2cdf>(const Packet2cdf& a)
{
  return pfirst(Packet2cdf(_mm256_add_ps(a.v.v, _mm256_permute2f128_ps(a.v.v, a.v.v, 1))));
}

template<> EIGEN_STRONG_INLINE Packet2cdf preduxp<Packet2cdf>(const Packet2cdf* vecs)
{
  return vecs[0];
}

template<> EIGEN_STRONG_INLINE std::complex<duals::dual<float> > predux_mul<Packet2cdf>(const Packet2cdf& a)
{
  return pfirst(pmul(a, preverse(a)));
}

#if 0
template<int Offset>
struct palign_impl<Offset,Packet2cdf>
{
  static EIGEN_STRONG_INLINE void run(Packet2cdf& /*first*/, const Packet2cdf& /*second*/)
  {
    // FIXME is it sure we never have to align a Packet2cdf?
    // Even though a std::complex<duals::dual<float> > has 32 bytes, it is not necessarily aligned on a 32 byte boundary...
  }
};
#endif

template<> struct conj_helper<Packet2cdf, Packet2cdf, false,true>
{
  EIGEN_STRONG_INLINE Packet2cdf pmadd(const Packet2cdf& x, const Packet2cdf& y, const Packet2cdf& c) const
  { return padd(pmul(x,y),c); }

  EIGEN_STRONG_INLINE Packet2cdf pmul(const Packet2cdf& a, const Packet2cdf& b) const
  {
    //#ifdef EIGEN_VECTORIZE_SSE3
    return internal::pmul(a, pconj(b));
    //#else
    // TODO for AVX
    //const __m128d mask = _mm_castsi128_pd(_mm_set_epi32(0x80000000,0x0,0x0,0x0));
    //return Packet2cdf(_mm_add_pd(_mm_xor_pd(_mm_mul_pd(vec2d_swizzle1(a.v.v, 0, 0), b.v), mask),
    //                            _mm_mul_pd(vec2d_swizzle1(a.v, 1, 1),
    //                                       vec2d_swizzle1(b.v, 1, 0))));
    //#endif
  }
};

template<> struct conj_helper<Packet2cdf, Packet2cdf, true,false>
{
  EIGEN_STRONG_INLINE Packet2cdf pmadd(const Packet2cdf& x, const Packet2cdf& y, const Packet2cdf& c) const
  { return padd(pmul(x,y),c); }

  EIGEN_STRONG_INLINE Packet2cdf pmul(const Packet2cdf& a, const Packet2cdf& b) const
  {
    //#ifdef EIGEN_VECTORIZE_SSE3
    return internal::pmul(pconj(a), b);
    //#else
    //const __m128d mask = _mm_castsi128_pd(_mm_set_epi32(0x80000000,0x0,0x0,0x0));
    //return Packet2cdf(_mm_add_pd(_mm_mul_pd(vec2d_swizzle1(a.v, 0, 0), b.v),
    //                            _mm_xor_pd(_mm_mul_pd(vec2d_swizzle1(a.v, 1, 1),
    //                                                  vec2d_swizzle1(b.v, 1, 0)), mask)));
    //#endif
  }
};

template<> struct conj_helper<Packet2cdf, Packet2cdf, true,true>
{
  EIGEN_STRONG_INLINE Packet2cdf pmadd(const Packet2cdf& x, const Packet2cdf& y, const Packet2cdf& c) const
  { return padd(pmul(x,y),c); }

  EIGEN_STRONG_INLINE Packet2cdf pmul(const Packet2cdf& a, const Packet2cdf& b) const
  {
    //#ifdef EIGEN_VECTORIZE_SSE3
    return pconj(internal::pmul(a, b));
    //#else
    //const __m128d mask = _mm_castsi128_pd(_mm_set_epi32(0x80000000,0x0,0x0,0x0));
    //return Packet2cdf(_mm_sub_pd(_mm_xor_pd(_mm_mul_pd(vec2d_swizzle1(a.v, 0, 0), b.v), mask),
    //                            _mm_mul_pd(vec2d_swizzle1(a.v, 1, 1),
    //                                       vec2d_swizzle1(b.v, 1, 0))));
    //#endif
  }
};

//TODO
EIGEN_MAKE_CONJ_HELPER_CPLX_REAL(Packet2cdf,Packet4df)

template<> EIGEN_STRONG_INLINE Packet2cdf pdiv<Packet2cdf>(const Packet2cdf& a, const Packet2cdf& b)
{
  Packet2cdf res = conj_helper<Packet2cdf,Packet2cdf,false,true>().pmul(a,b);
  Packet4df s = pmul(b.v, b.v);
  return Packet2cdf(pdiv(res.v,
                         padd(s, Packet4df(_mm256_castpd_ps(_mm256_shuffle_pd(_mm256_castps_pd(s.v),
                                                                              _mm256_castps_pd(s.v), 0x5))))));
}

EIGEN_STRONG_INLINE Packet2cdf pcplxflip/* <Packet2cdf> */(const Packet2cdf& x)
{
  return Packet2cdf(vec8f_swizzle1(x.v.v, 2, 3, 0, 1));
}

EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<Packet2cdf,2> & kernel)
{
  __m256d tmp = _mm256_permute2f128_pd(_mm256_castps_pd(kernel.packet[0].v.v),
                                       _mm256_castps_pd(kernel.packet[1].v.v), 0+(2<<4));
  kernel.packet[1].v.v = _mm256_castpd_ps(_mm256_permute2f128_pd(_mm256_castps_pd(kernel.packet[0].v.v),
                                                                 _mm256_castps_pd(kernel.packet[1].v.v), 1+(3<<4)));
  kernel.packet[0].v.v = _mm256_castpd_ps(tmp);
}

//---------- double ----------
#ifdef __AVX2__

struct Packet1cdd
{
  EIGEN_STRONG_INLINE Packet1cdd() {}
  EIGEN_STRONG_INLINE explicit Packet1cdd(const __m256d & a) : v(a) {}
  EIGEN_STRONG_INLINE explicit Packet1cdd(const Packet2dd & a) : v(a) {}
  //__m256d  v;
  Packet2dd  v;
};

#if !defined(CPPDUALS_DONT_VECTORIZE_CDUAL)
template<> struct packet_traits<std::complex<duals::dual<double> > >  : default_packet_traits
{
  typedef Packet1cdd type;
  typedef Packet1cdd half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 0,
    size = 1,
    HasHalfPacket = 0,

    HasAdd    = 1,
    HasSub    = 1,
    HasMul    = 1,
    HasDiv    = 1,
    HasNegate = 1,
    HasAbs    = 0,
    HasAbs2   = 0,
    HasMin    = 0,
    HasMax    = 0,
    HasSetLinear = 0
  };
};
#endif

template<> struct unpacket_traits<Packet1cdd> {
  typedef std::complex<duals::dual<double> > type;
  enum {size=1, alignment=Aligned32, masked_load_available=false, masked_store_available=false, vectorizable=true};
  typedef Packet1cdd half;
};

template<> EIGEN_STRONG_INLINE Packet1cdd padd<Packet1cdd>(const Packet1cdd& a, const Packet1cdd& b)
{ return Packet1cdd(_mm256_add_pd(a.v.v,b.v.v)); }
template<> EIGEN_STRONG_INLINE Packet1cdd psub<Packet1cdd>(const Packet1cdd& a, const Packet1cdd& b)
{ return Packet1cdd(_mm256_sub_pd(a.v.v,b.v.v)); }
template<> EIGEN_STRONG_INLINE Packet1cdd pnegate(const Packet1cdd& a) { return Packet1cdd(pnegate(a.v.v)); }
template<> EIGEN_STRONG_INLINE Packet1cdd pconj(const Packet1cdd& a)
{
  const __m256d mask = _mm256_castsi256_pd(_mm256_set_epi32(0x80000000,0x0,0x80000000,0x0,0x0,0x0,0x0,0x0));
  return Packet1cdd(_mm256_xor_pd(a.v.v,mask));
}

template<> EIGEN_STRONG_INLINE Packet1cdd pmul<Packet1cdd>(const Packet1cdd& a, const Packet1cdd& b)
{
#ifndef __AVX2__
  // TODO
#error "no avx2, really?"
#endif
#ifdef __FMA__
  const __m256d mask = _mm256_castsi256_pd(_mm256_setr_epi32(0x0,0x0,0x0,0x0,
                                                             0xffffffff,0xffffffff,0xffffffff,0xffffffff));
  return Packet1cdd(vec4d_swizzle1
                    (_mm256_addsub_pd(_mm256_fmadd_pd(_mm256_broadcastsd_pd(_mm256_castpd256_pd128(a.v.v)),
                                                      vec4d_swizzle1(b.v.v, 0, 2, 1, 3),
                                                      _mm256_and_pd(mask,
                                                                    _mm256_mul_pd(vec4d_swizzle1(a.v.v, 0, 0, 1, 1),
                                                                                  vec4d_swizzle1(b.v.v, 0, 0, 0, 2)))),
                                      _mm256_fmadd_pd(vec4d_swizzle1(a.v.v, 2, 2, 2, 2),
                                                      vec4d_swizzle1(b.v.v, 2, 0, 3, 1),
                                                      _mm256_and_pd(mask,
                                                                    _mm256_mul_pd(vec4d_swizzle1(a.v.v, 0, 0, 3, 3),
                                                                                  vec4d_swizzle1(b.v.v, 0, 0, 2, 0))))),
                     0,2,1,3));
#else
  const __m256d mask = _mm256_castsi256_pd(_mm256_setr_epi32(0x0,0x0,0x0,0x0,
                                                             0xffffffff,0xffffffff,0xffffffff,0xffffffff));
  return Packet1cdd(vec4d_swizzle1
                    (_mm256_addsub_pd(_mm256_add_pd(_mm256_mul_pd(_mm256_broadcastsd_pd(_mm256_castpd256_pd128(a.v.v)),
                                                                  vec4d_swizzle1(b.v.v, 0, 2, 1, 3)),
                                                    _mm256_and_pd(mask,
                                                                  _mm256_mul_pd(vec4d_swizzle1(a.v.v, 0, 0, 1, 1),
                                                                                vec4d_swizzle1(b.v.v, 0, 0, 0, 2)))),
                                      _mm256_add_pd(_mm256_mul_pd(vec4d_swizzle1(a.v.v, 2, 2, 2, 2),
                                                                  vec4d_swizzle1(b.v.v, 2, 0, 3, 1)),
                                                    _mm256_and_pd(mask,
                                                                  _mm256_mul_pd(vec4d_swizzle1(a.v.v, 0, 0, 3, 3),
                                                                                vec4d_swizzle1(b.v.v, 0, 0, 2, 0))))),
                     0,2,1,3));
#endif
}

template<> EIGEN_STRONG_INLINE Packet1cdd pand   <Packet1cdd>(const Packet1cdd& a, const Packet1cdd& b)
{ return Packet1cdd(_mm256_and_pd(a.v.v,b.v.v)); }
template<> EIGEN_STRONG_INLINE Packet1cdd por    <Packet1cdd>(const Packet1cdd& a, const Packet1cdd& b)
{ return Packet1cdd(_mm256_or_pd(a.v.v,b.v.v)); }
template<> EIGEN_STRONG_INLINE Packet1cdd pxor   <Packet1cdd>(const Packet1cdd& a, const Packet1cdd& b)
{ return Packet1cdd(_mm256_xor_pd(a.v.v,b.v.v)); }
template<> EIGEN_STRONG_INLINE Packet1cdd pandnot<Packet1cdd>(const Packet1cdd& a, const Packet1cdd& b)
{ return Packet1cdd(_mm256_andnot_pd(a.v.v,b.v.v)); }

template<> EIGEN_STRONG_INLINE Packet1cdd pload <Packet1cdd>(const std::complex<duals::dual<double> >* from)
{ EIGEN_DEBUG_ALIGNED_LOAD return Packet1cdd(ploadu<Packet4d>((const double*)from)); }
template<> EIGEN_STRONG_INLINE Packet1cdd ploadu<Packet1cdd>(const std::complex<duals::dual<double> >* from)
{ EIGEN_DEBUG_UNALIGNED_LOAD return Packet1cdd(ploadu<Packet4d>((const double*)from)); }
template<> EIGEN_STRONG_INLINE Packet1cdd pset1<Packet1cdd>(const std::complex<duals::dual<double> >&  from)
{ /* here we really have to use unaligned loads :( */ return ploadu<Packet1cdd>(&from); }

template<> EIGEN_STRONG_INLINE Packet1cdd ploaddup<Packet1cdd>(const std::complex<duals::dual<double> >* from)
{ return pset1<Packet1cdd>(*from); }

// FIXME force unaligned store, this is a temporary fix
template<> EIGEN_STRONG_INLINE void
pstore <std::complex<duals::dual<double> > >(std::complex<duals::dual<double> > *   to, const Packet1cdd& from)
{ EIGEN_DEBUG_ALIGNED_STORE pstoreu((double*)to, from.v.v); }
template<> EIGEN_STRONG_INLINE void
pstoreu<std::complex<duals::dual<double> > >(std::complex<duals::dual<double> > *   to, const Packet1cdd& from)
{ EIGEN_DEBUG_UNALIGNED_STORE pstoreu((double*)to, from.v.v); }

template<> EIGEN_STRONG_INLINE void
prefetch<std::complex<duals::dual<double> > >(const std::complex<duals::dual<double> > *   addr)
{ _mm_prefetch((SsePrefetchPtrType)(addr), _MM_HINT_T0); }

template<> EIGEN_STRONG_INLINE std::complex<duals::dual<double> >  pfirst<Packet1cdd>(const Packet1cdd& a)
{
  EIGEN_ALIGN16 double res[4];
  _mm256_store_pd(res, a.v.v);
  return std::complex<duals::dual<double> >(duals::dual<double>(res[0],res[1]),
                                           duals::dual<double>(res[2],res[3]));
}

template<> EIGEN_STRONG_INLINE Packet1cdd preverse(const Packet1cdd& a) { return a; }

template<> EIGEN_STRONG_INLINE std::complex<duals::dual<double> > predux<Packet1cdd>(const Packet1cdd& a)
{
  return pfirst(a);
}

template<> EIGEN_STRONG_INLINE Packet1cdd preduxp<Packet1cdd>(const Packet1cdd* vecs)
{
  return vecs[0];
}

template<> EIGEN_STRONG_INLINE std::complex<duals::dual<double> > predux_mul<Packet1cdd>(const Packet1cdd& a)
{
  return pfirst(a);
}

template<int Offset>
struct palign_impl<Offset,Packet1cdd>
{
  static EIGEN_STRONG_INLINE void run(Packet1cdd& /*first*/, const Packet1cdd& /*second*/)
  {
    // FIXME is it sure we never have to align a Packet1cdd?
    // Even though a std::complex<duals::dual<double> > has 16 bytes, it is not necessarily aligned on a 16 bytes boundary...
  }
};

template<> struct conj_helper<Packet1cdd, Packet1cdd, false,true>
{
  EIGEN_STRONG_INLINE Packet1cdd pmadd(const Packet1cdd& x, const Packet1cdd& y, const Packet1cdd& c) const
  { return padd(pmul(x,y),c); }

  EIGEN_STRONG_INLINE Packet1cdd pmul(const Packet1cdd& a, const Packet1cdd& b) const
  {
    return internal::pmul(a, pconj(b));
  }
};

template<> struct conj_helper<Packet1cdd, Packet1cdd, true,false>
{
  EIGEN_STRONG_INLINE Packet1cdd pmadd(const Packet1cdd& x, const Packet1cdd& y, const Packet1cdd& c) const
  { return padd(pmul(x,y),c); }

  EIGEN_STRONG_INLINE Packet1cdd pmul(const Packet1cdd& a, const Packet1cdd& b) const
  {
    return internal::pmul(pconj(a), b);
  }
};

template<> struct conj_helper<Packet1cdd, Packet1cdd, true,true>
{
  EIGEN_STRONG_INLINE Packet1cdd pmadd(const Packet1cdd& x, const Packet1cdd& y, const Packet1cdd& c) const
  { return padd(pmul(x,y),c); }

  EIGEN_STRONG_INLINE Packet1cdd pmul(const Packet1cdd& a, const Packet1cdd& b) const
  {
    return pconj(internal::pmul(a, b));
  }
};

EIGEN_MAKE_CONJ_HELPER_CPLX_REAL(Packet1cdd,Packet2dd)

template<> EIGEN_STRONG_INLINE Packet1cdd pdiv<Packet1cdd>(const Packet1cdd& a, const Packet1cdd& b)
{
  //Packet2cd num = pmul(a, pconj(b));
  //__m256d tmp = _mm256_mul_pd(b.v, b.v);
  //__m256d denom = _mm256_hadd_pd(tmp, tmp);
  //return Packet2cd(_mm256_div_pd(num.v, denom));

  Packet1cdd num = conj_helper<Packet1cdd,Packet1cdd,false,true>().pmul(a,b);
  Packet2dd tmp = pmul(b.v, b.v);
  Packet2dd denom ( _mm256_add_pd(tmp.v,
                                  _mm256_permute2f128_pd(tmp.v,tmp.v, 1)));
  return Packet1cdd(pdiv(num.v, denom));
}

EIGEN_STRONG_INLINE Packet1cdd pcplxflip/* <Packet1cdd> */(const Packet1cdd& x)
{
  return Packet1cdd(_mm256_permute2f128_pd(x.v.v,x.v.v, 1));
}
#else
#warning "AVX2 disabled: not vectorizing std::complex<dual<double>>"
#endif

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_CDUAL_AVX_H

// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2019 Michael Tesch <tesch1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CDUAL_SSE_H
#define EIGEN_CDUAL_SSE_H

namespace Eigen {

namespace internal {

//---------- float ----------
struct Packet1cdf
{
  EIGEN_STRONG_INLINE Packet1cdf() {}
  EIGEN_STRONG_INLINE explicit Packet1cdf(const __m128 & a) : v(a) {}
  EIGEN_STRONG_INLINE explicit Packet1cdf(const Packet2df & a) : v(a) {}
  //__m128  v;
  Packet2df  v;
};

// Use the packet_traits defined in AVX/ComplexDual.h instead if we're
// going to leverage AVX instructions.
#if !defined(CPPDUALS_DONT_VECTORIZE_CDUAL)
#if !defined(EIGEN_VECTORIZE_AVX)
template<> struct packet_traits<std::complex<duals::dual<float> > >  : default_packet_traits
{
  typedef Packet1cdf type;
  typedef Packet1cdf half;
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
#endif

template<> struct unpacket_traits<Packet1cdf> {
  typedef std::complex<duals::dual<float> > type;
  enum {size=1, alignment=Aligned16, masked_load_available=false, masked_store_available=false, vectorizable=true};
  typedef Packet1cdf half;
};

template<> EIGEN_STRONG_INLINE Packet1cdf padd<Packet1cdf>(const Packet1cdf& a, const Packet1cdf& b)
{ return Packet1cdf(_mm_add_ps(a.v.v,b.v.v)); }
template<> EIGEN_STRONG_INLINE Packet1cdf psub<Packet1cdf>(const Packet1cdf& a, const Packet1cdf& b)
{ return Packet1cdf(_mm_sub_ps(a.v.v,b.v.v)); }
template<> EIGEN_STRONG_INLINE Packet1cdf pnegate(const Packet1cdf& a) { return Packet1cdf(pnegate(a.v)); }
template<> EIGEN_STRONG_INLINE Packet1cdf pconj(const Packet1cdf& a)
{
  const __m128 mask = _mm_castsi128_ps(_mm_set_epi32(0x80000000,0x80000000,0x0,0x0));
  return Packet1cdf(_mm_xor_ps(a.v.v,mask));
}

template<> EIGEN_STRONG_INLINE Packet1cdf pmul<Packet1cdf>(const Packet1cdf& a, const Packet1cdf& b)
{
#if 1 // defined(EIGEN_VECTORIZE_SSE3)
  const __m128 mask = _mm_castsi128_ps(_mm_setr_epi32(0x00000000,0x00000000,0xffffffff,0xffffffff));
  return Packet1cdf(vec4f_swizzle1
                    (_mm_addsub_ps(_mm_add_ps(_mm_mul_ps(vec4f_swizzle1(a.v.v, 0, 0, 0, 0),
                                                         vec4f_swizzle1(b.v.v, 0, 2, 1, 3)),
                                              _mm_and_ps(mask,
                                                         _mm_mul_ps(vec4f_swizzle1(a.v.v, 0, 0, 1, 1),
                                                                    vec4f_swizzle1(b.v.v, 0, 0, 0, 2)))),
                                   _mm_add_ps(_mm_mul_ps(vec4f_swizzle1(a.v.v, 2, 2, 2, 2),
                                                         vec4f_swizzle1(b.v.v, 2, 0, 3, 1)),
                                              _mm_and_ps(mask,
                                                         _mm_mul_ps(vec4f_swizzle1(a.v.v, 0, 0, 3, 3),
                                                                    vec4f_swizzle1(b.v.v, 0, 0, 2, 0))))),
                     0,2,1,3));
#else
  const __m128 mask = _mm_castsi128_ps(_mm_setr_epi32(0x00000000,0xffffffff,0x00000000,0xffffffff));
  const __m128 nega = _mm_castsi128_ps(_mm_setr_epi32(0x80000000,0x80000000,0x00000000,0x00000000));
  return Packet1cdf(_mm_add_ps(_mm_add_ps(_mm_mul_ps(vec4f_swizzle1(a.v.v, 0, 0, 0, 0),
                                                     vec4f_swizzle1(b.v.v, 0, 1, 2, 3)),
                                          _mm_and_ps(mask,
                                                     _mm_mul_ps(vec4f_swizzle1(a.v.v, 0, 1, 0, 1),
                                                                vec4f_swizzle1(b.v.v, 0, 0, 0, 2)))),
                               _mm_xor_ps
                               (nega,
                                _mm_add_ps(_mm_mul_ps(vec4f_swizzle1(a.v.v, 2, 2, 2, 2),
                                                      vec4f_swizzle1(b.v.v, 2, 3, 0, 1)),
                                           _mm_and_ps(mask,
                                                      _mm_mul_ps(vec4f_swizzle1(a.v.v, 0, 3, 0, 3),
                                                                 vec4f_swizzle1(b.v.v, 0, 2, 0, 0)))))));
#endif
  return a;
}

template<> EIGEN_STRONG_INLINE Packet1cdf pand   <Packet1cdf>(const Packet1cdf& a, const Packet1cdf& b)
{ return Packet1cdf(_mm_and_ps(a.v.v,b.v.v)); }
template<> EIGEN_STRONG_INLINE Packet1cdf por    <Packet1cdf>(const Packet1cdf& a, const Packet1cdf& b)
{ return Packet1cdf(_mm_or_ps(a.v.v,b.v.v)); }
template<> EIGEN_STRONG_INLINE Packet1cdf pxor   <Packet1cdf>(const Packet1cdf& a, const Packet1cdf& b)
{ return Packet1cdf(_mm_xor_ps(a.v.v,b.v.v)); }
template<> EIGEN_STRONG_INLINE Packet1cdf pandnot<Packet1cdf>(const Packet1cdf& a, const Packet1cdf& b)
{ return Packet1cdf(_mm_andnot_ps(a.v.v,b.v.v)); }

template<> EIGEN_STRONG_INLINE Packet1cdf pload <Packet1cdf>(const std::complex<duals::dual<float> >* from)
{ EIGEN_DEBUG_ALIGNED_LOAD return Packet1cdf(ploadu<Packet4f>((const float*)from)); }
template<> EIGEN_STRONG_INLINE Packet1cdf ploadu<Packet1cdf>(const std::complex<duals::dual<float> >* from)
{ EIGEN_DEBUG_UNALIGNED_LOAD return Packet1cdf(ploadu<Packet4f>((const float*)from)); }
template<> EIGEN_STRONG_INLINE Packet1cdf pset1<Packet1cdf>(const std::complex<duals::dual<float> >&  from)
{ /* here we really have to use unaligned loads :( */ return ploadu<Packet1cdf>(&from); }

template<> EIGEN_STRONG_INLINE Packet1cdf ploaddup<Packet1cdf>(const std::complex<duals::dual<float> >* from)
{ return pset1<Packet1cdf>(*from); }

// FIXME force unaligned store, this is a temporary fix
template<> EIGEN_STRONG_INLINE void
pstore <std::complex<duals::dual<float> > >(std::complex<duals::dual<float> > *   to, const Packet1cdf& from)
{ EIGEN_DEBUG_ALIGNED_STORE pstoreu((float*)to, from.v.v); }
template<> EIGEN_STRONG_INLINE void
pstoreu<std::complex<duals::dual<float> > >(std::complex<duals::dual<float> > *   to, const Packet1cdf& from)
{ EIGEN_DEBUG_UNALIGNED_STORE pstoreu((float*)to, from.v.v); }

template<> EIGEN_STRONG_INLINE void
prefetch<std::complex<duals::dual<float> > >(const std::complex<duals::dual<float> > *   addr)
{ _mm_prefetch((SsePrefetchPtrType)(addr), _MM_HINT_T0); }

template<> EIGEN_STRONG_INLINE std::complex<duals::dual<float> >  pfirst<Packet1cdf>(const Packet1cdf& a)
{
  EIGEN_ALIGN16 float res[4];
  _mm_store_ps(res, a.v.v);
  return std::complex<duals::dual<float> >(duals::dual<float>(res[0],res[1]),
                                           duals::dual<float>(res[2],res[3]));
}

template<> EIGEN_STRONG_INLINE Packet1cdf preverse(const Packet1cdf& a) { return a; }

template<> EIGEN_STRONG_INLINE std::complex<duals::dual<float> > predux<Packet1cdf>(const Packet1cdf& a)
{
  return pfirst(a);
}

template<> EIGEN_STRONG_INLINE Packet1cdf preduxp<Packet1cdf>(const Packet1cdf* vecs)
{
  return vecs[0];
}

template<> EIGEN_STRONG_INLINE std::complex<duals::dual<float> > predux_mul<Packet1cdf>(const Packet1cdf& a)
{
  return pfirst(a);
}

template<int Offset>
struct palign_impl<Offset,Packet1cdf>
{
  static EIGEN_STRONG_INLINE void run(Packet1cdf& /*first*/, const Packet1cdf& /*second*/)
  {
    // FIXME is it sure we never have to align a Packet1cdf?
    // Even though a std::complex<duals::dual<float> > has 16 bytes, it is not necessarily aligned on a 16 bytes boundary...
  }
};

template<> struct conj_helper<Packet1cdf, Packet1cdf, false,true>
{
  EIGEN_STRONG_INLINE Packet1cdf pmadd(const Packet1cdf& x, const Packet1cdf& y, const Packet1cdf& c) const
  { return padd(pmul(x,y),c); }

  EIGEN_STRONG_INLINE Packet1cdf pmul(const Packet1cdf& a, const Packet1cdf& b) const
  {
    return internal::pmul(a, pconj(b));
  }
};

template<> struct conj_helper<Packet1cdf, Packet1cdf, true,false>
{
  EIGEN_STRONG_INLINE Packet1cdf pmadd(const Packet1cdf& x, const Packet1cdf& y, const Packet1cdf& c) const
  { return padd(pmul(x,y),c); }

  EIGEN_STRONG_INLINE Packet1cdf pmul(const Packet1cdf& a, const Packet1cdf& b) const
  {
    return internal::pmul(pconj(a), b);
  }
};

template<> struct conj_helper<Packet1cdf, Packet1cdf, true,true>
{
  EIGEN_STRONG_INLINE Packet1cdf pmadd(const Packet1cdf& x, const Packet1cdf& y, const Packet1cdf& c) const
  { return padd(pmul(x,y),c); }

  EIGEN_STRONG_INLINE Packet1cdf pmul(const Packet1cdf& a, const Packet1cdf& b) const
  {
    return pconj(internal::pmul(a, b));
  }
};

//TODO
EIGEN_MAKE_CONJ_HELPER_CPLX_REAL(Packet1cdf,Packet2df)

template<> EIGEN_STRONG_INLINE Packet1cdf pdiv<Packet1cdf>(const Packet1cdf& a, const Packet1cdf& b)
{
  Packet1cdf res = conj_helper<Packet1cdf,Packet1cdf,false,true>().pmul(a,b);
  Packet2df s = pmul(b.v, b.v);
  return Packet1cdf(pdiv(res.v,
                         padd(s, Packet2df(_mm_castpd_ps(_mm_shuffle_pd(_mm_castps_pd(s.v),
                                                                        _mm_castps_pd(s.v), 0x1))))));
}

EIGEN_STRONG_INLINE Packet1cdf pcplxflip/* <Packet1cdf> */(const Packet1cdf& x)
{
  return Packet1cdf(vec4f_swizzle1(x.v.v, 2, 3, 0, 1));
}

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_CDUAL_SSE_H

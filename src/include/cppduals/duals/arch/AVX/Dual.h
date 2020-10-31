// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner (benoit.steiner.goog@gmail.com)
// Copyright (C) 2019 Michael Tesch <tesch1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_DUAL_AVX_H
#define EIGEN_DUAL_AVX_H

namespace Eigen {

namespace internal {

//---------- float ----------
struct Packet4df
{
  EIGEN_STRONG_INLINE Packet4df() {}
  EIGEN_STRONG_INLINE explicit Packet4df(const __m256& a) : v(a) {}
  __m256  v;
};

template<> struct packet_traits<duals::dual<float> >  : default_packet_traits
{
  typedef Packet4df type;
  typedef Packet2df half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 4,
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

template<> struct unpacket_traits<Packet4df> {
  typedef duals::dual<float> type;
  enum {size=4, alignment=Aligned32, masked_load_available=false, masked_store_available=false, vectorizable=true};
  typedef Packet2df half;
};

//template<> EIGEN_STRONG_INLINE Packet4df pzero(const Packet4df& /*a*/) { return Packet4df(_mm256_setzero_ps()); }
template<> EIGEN_STRONG_INLINE Packet4df padd<Packet4df>(const Packet4df& a, const Packet4df& b)
{ return Packet4df(_mm256_add_ps(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet4df psub<Packet4df>(const Packet4df& a, const Packet4df& b)
{ return Packet4df(_mm256_sub_ps(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet4df pnegate(const Packet4df& a)
{
  return Packet4df(pnegate(a.v));
}
template<> EIGEN_STRONG_INLINE Packet4df pdconj(const Packet4df& a)
{
  const __m256 mask = _mm256_castsi256_ps(_mm256_setr_epi32(0x00000000,0x80000000,0x00000000,0x80000000,
                                                            0x00000000,0x80000000,0x00000000,0x80000000));
  return Packet4df(_mm256_xor_ps(a.v,mask));
}
template<> EIGEN_STRONG_INLINE Packet4df pconj(const Packet4df& a) { return a; }

template<> EIGEN_STRONG_INLINE Packet4df pmul<Packet4df>(const Packet4df& a, const Packet4df& b)
{
#ifdef xx__FMA__
  __m256 result = _mm256_fmadd_ps(_mm256_moveldup_ps(a.v),
                                  b.v,
                                  _mm256_blend_ps(_mm256_setzero_ps(),
                                                  _mm256_mul_ps(_mm256_moveldup_ps(b.v), a.v), 0xaa));
  return Packet4df(result);
#else
  __m256 result = _mm256_add_ps(_mm256_mul_ps(_mm256_moveldup_ps(a.v), b.v),
                                _mm256_blend_ps(_mm256_setzero_ps(),
                                                _mm256_mul_ps(_mm256_moveldup_ps(b.v), a.v), 0xaa));
#endif
  return Packet4df(result);
}

template<> EIGEN_STRONG_INLINE Packet4df pand   <Packet4df>(const Packet4df& a, const Packet4df& b)
{ return Packet4df(_mm256_and_ps(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet4df por    <Packet4df>(const Packet4df& a, const Packet4df& b)
{ return Packet4df(_mm256_or_ps(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet4df pxor   <Packet4df>(const Packet4df& a, const Packet4df& b)
{ return Packet4df(_mm256_xor_ps(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet4df pandnot<Packet4df>(const Packet4df& a, const Packet4df& b)
{ return Packet4df(_mm256_andnot_ps(a.v,b.v)); }

template<> EIGEN_STRONG_INLINE Packet4df pload1<Packet4df>(const duals::dual<float>* from)
{
  return Packet4df(_mm256_castpd_ps(_mm256_broadcast_sd((const double*)(const void*)from)));
}
template<> EIGEN_STRONG_INLINE Packet4df pload <Packet4df>(const duals::dual<float>* from)
{ EIGEN_DEBUG_ALIGNED_LOAD return Packet4df(pload<Packet8f>(&numext::real_ref(*from))); }
template<> EIGEN_STRONG_INLINE Packet4df ploadu<Packet4df>(const duals::dual<float>* from)
{ EIGEN_DEBUG_UNALIGNED_LOAD return Packet4df(ploadu<Packet8f>(&numext::real_ref(*from))); }


template<> EIGEN_STRONG_INLINE Packet4df pset1<Packet4df>(const duals::dual<float>& from)
{
  return Packet4df(_mm256_castpd_ps(_mm256_broadcast_sd((const double*)(const void*)&from)));
}

template<> EIGEN_STRONG_INLINE Packet4df ploaddup<Packet4df>(const duals::dual<float>* from)
{
  // FIXME The following might be optimized using _mm256_movedup_pd
  Packet2df a = ploaddup<Packet2df>(from);
  Packet2df b = ploaddup<Packet2df>(from+1);
  return  Packet4df(_mm256_insertf128_ps(_mm256_castps128_ps256(a.v), b.v, 1));
}

template<> EIGEN_STRONG_INLINE void pstore <duals::dual<float> >(duals::dual<float>* to, const Packet4df& from)
{ EIGEN_DEBUG_ALIGNED_STORE pstore(&numext::real_ref(*to), from.v); }
template<> EIGEN_STRONG_INLINE void pstoreu<duals::dual<float> >(duals::dual<float>* to, const Packet4df& from)
{ EIGEN_DEBUG_UNALIGNED_STORE pstoreu(&numext::real_ref(*to), from.v); }

template<> EIGEN_DEVICE_FUNC inline Packet4df pgather<duals::dual<float>, Packet4df>(const duals::dual<float>* from,
                                                                                     Index stride)
{
  return Packet4df(_mm256_set_ps(duals::dpart(from[3*stride]), duals::rpart(from[3*stride]),
                                 duals::dpart(from[2*stride]), duals::rpart(from[2*stride]),
                                 duals::dpart(from[1*stride]), duals::rpart(from[1*stride]),
                                 duals::dpart(from[0*stride]), duals::rpart(from[0*stride])));
}

template<> EIGEN_DEVICE_FUNC inline void pscatter<duals::dual<float>, Packet4df>(duals::dual<float>* to,
                                                                                 const Packet4df& from, Index stride)
{
  __m128 low = _mm256_extractf128_ps(from.v, 0);
  to[stride*0] = duals::dual<float>(_mm_cvtss_f32(_mm_shuffle_ps(low, low, 0)),
                                     _mm_cvtss_f32(_mm_shuffle_ps(low, low, 1)));
  to[stride*1] = duals::dual<float>(_mm_cvtss_f32(_mm_shuffle_ps(low, low, 2)),
                                     _mm_cvtss_f32(_mm_shuffle_ps(low, low, 3)));

  __m128 high = _mm256_extractf128_ps(from.v, 1);
  to[stride*2] = duals::dual<float>(_mm_cvtss_f32(_mm_shuffle_ps(high, high, 0)),
                                     _mm_cvtss_f32(_mm_shuffle_ps(high, high, 1)));
  to[stride*3] = duals::dual<float>(_mm_cvtss_f32(_mm_shuffle_ps(high, high, 2)),
                                     _mm_cvtss_f32(_mm_shuffle_ps(high, high, 3)));

}

template<> EIGEN_STRONG_INLINE duals::dual<float>  pfirst<Packet4df>(const Packet4df& a)
{
  return pfirst(Packet2df(_mm256_castps256_ps128(a.v)));
}

template<> EIGEN_STRONG_INLINE Packet4df preverse(const Packet4df& a) {
  __m128 low  = _mm256_extractf128_ps(a.v, 0);
  __m128 high = _mm256_extractf128_ps(a.v, 1);
  __m128d lowd  = _mm_castps_pd(low);
  __m128d highd = _mm_castps_pd(high);
  low  = _mm_castpd_ps(_mm_shuffle_pd(lowd,lowd,0x1));
  high = _mm_castpd_ps(_mm_shuffle_pd(highd,highd,0x1));
  __m256 result = _mm256_setzero_ps();
  result = _mm256_insertf128_ps(result, low, 1);
  result = _mm256_insertf128_ps(result, high, 0);
  return Packet4df(result);
}

template<> EIGEN_STRONG_INLINE duals::dual<float> predux<Packet4df>(const Packet4df& a)
{
  return predux(padd(Packet2df(_mm256_extractf128_ps(a.v,0)),
                     Packet2df(_mm256_extractf128_ps(a.v,1))));
}

template<> EIGEN_STRONG_INLINE Packet4df preduxp<Packet4df>(const Packet4df* vecs)
{
  Packet8f t0 = _mm256_shuffle_ps(vecs[0].v, vecs[0].v, _MM_SHUFFLE(3, 1, 2 ,0));
  Packet8f t1 = _mm256_shuffle_ps(vecs[1].v, vecs[1].v, _MM_SHUFFLE(3, 1, 2 ,0));
  t0 = _mm256_hadd_ps(t0,t1);
  Packet8f t2 = _mm256_shuffle_ps(vecs[2].v, vecs[2].v, _MM_SHUFFLE(3, 1, 2 ,0));
  Packet8f t3 = _mm256_shuffle_ps(vecs[3].v, vecs[3].v, _MM_SHUFFLE(3, 1, 2 ,0));
  t2 = _mm256_hadd_ps(t2,t3);

  t1 = _mm256_permute2f128_ps(t0,t2, 0 + (2<<4));
  t3 = _mm256_permute2f128_ps(t0,t2, 1 + (3<<4));

  return Packet4df(_mm256_add_ps(t1,t3));
}

template<> EIGEN_STRONG_INLINE duals::dual<float> predux_mul<Packet4df>(const Packet4df& a)
{
  return predux_mul(pmul(Packet2df(_mm256_extractf128_ps(a.v, 0)),
                         Packet2df(_mm256_extractf128_ps(a.v, 1))));
}

template<int Offset>
struct palign_impl<Offset,Packet4df>
{
  static EIGEN_STRONG_INLINE void run(Packet4df& first, const Packet4df& second)
  {
    if (Offset==0) return;
    palign_impl<Offset*2,Packet8f>::run(first.v, second.v);
  }
};

#if 0 // TODO
template<> struct dconj_helper<Packet4df, Packet4df, false,true>
{
  EIGEN_STRONG_INLINE Packet4df pmadd(const Packet4df& x, const Packet4df& y, const Packet4df& c) const
  { return padd(pmul(x,y),c); }
  EIGEN_STRONG_INLINE Packet4df pmul(const Packet4df& a, const Packet4df& b) const
  { return internal::pmul(a, pdconj(b)); }
};

template<> struct dconj_helper<Packet4df, Packet4df, true,false>
{
  EIGEN_STRONG_INLINE Packet4df pmadd(const Packet4df& x, const Packet4df& y, const Packet4df& c) const
  { return padd(pmul(x,y),c); }
  EIGEN_STRONG_INLINE Packet4df pmul(const Packet4df& a, const Packet4df& b) const
  { return internal::pmul(pdconj(a), b); }
};

template<> struct dconj_helper<Packet4df, Packet4df, true,true>
{
  EIGEN_STRONG_INLINE Packet4df pmadd(const Packet4df& x, const Packet4df& y, const Packet4df& c) const
  { return padd(pmul(x,y),c); }
  EIGEN_STRONG_INLINE Packet4df pmul(const Packet4df& a, const Packet4df& b) const
  { return pdconj(internal::pmul(a, b)); }
};
EIGEN_MAKE_DCONJ_HELPER_DUAL_REAL(Packet4df,Packet8f)
#endif

template<> EIGEN_STRONG_INLINE Packet4df pdiv<Packet4df>(const Packet4df& a, const Packet4df& b)
{
#if 0
  // approxer, but faster?
  const __m256 mask = _mm256_castsi256_ps(_mm256_setr_epi32(0x00000000,0xffffffff,0x00000000,0xffffffff,
                                                            0x00000000,0xffffffff,0x00000000,0xffffffff));
  __m256 xr = _mm256_moveldup_ps(b.v);
  __m256 num = _mm256_sub_ps(_mm256_mul_ps(a.v, xr),
                             _mm256_and_ps(mask,
                                           _mm256_mul_ps(b.v, _mm256_moveldup_ps(a.v))));
  return Packet4df(_mm256_div_ps(num,
                                 _mm256_mul_ps(xr,xr)));
#else
  const __m256 mask = _mm256_castsi256_ps(_mm256_setr_epi32(0xffffffff,0x00000000,0xffffffff,0x00000000,
                                                            0xffffffff,0x00000000,0xffffffff,0x00000000));
  __m256 xr = _mm256_moveldup_ps(b.v);
  __m256 r = _mm256_div_ps(_mm256_add_ps(_mm256_and_ps(mask, a.v),
                                         _mm256_andnot_ps(mask,
                                                          _mm256_sub_ps(_mm256_mul_ps(a.v, xr),
                                                                        _mm256_mul_ps(b.v, _mm256_moveldup_ps(a.v))))),
                           _mm256_add_ps(_mm256_and_ps(mask, b.v),
                                         _mm256_andnot_ps(mask,
                                                          _mm256_mul_ps(xr,xr))));
  return Packet4df(r);
#endif
}

//---------- double ----------
struct Packet2dd
{
  EIGEN_STRONG_INLINE Packet2dd() {}
  EIGEN_STRONG_INLINE explicit Packet2dd(const __m256d& a) : v(a) {}
  __m256d  v;
};

template<> struct packet_traits<duals::dual<double> >  : default_packet_traits
{
  typedef Packet2dd type;
#if EIGEN_VERSION_AT_LEAST(3, 3, 8)
  // sometime after 3.3.7 gebp_traits cant deal with this
#define TWODD_NOHALF
#endif
#ifdef TWODD_NOHALF
  typedef Packet2dd half;
#else
  typedef Packet1dd half;
#endif
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 0,
    size = 2,
#ifdef TWODD_NOHALF
    HasHalfPacket = 0,
#else
    HasHalfPacket = 1,
#endif

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

template<> struct unpacket_traits<Packet2dd> {
  typedef duals::dual<double> type;
  enum {size=2, alignment=Aligned32, masked_load_available=false, masked_store_available=false, vectorizable=true};
#ifdef TWODD_NOHALF
  typedef Packet2dd half;
#else
  typedef Packet1dd half;
#endif
};

template<> EIGEN_STRONG_INLINE Packet2dd padd<Packet2dd>(const Packet2dd& a, const Packet2dd& b)
{ return Packet2dd(_mm256_add_pd(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet2dd psub<Packet2dd>(const Packet2dd& a, const Packet2dd& b)
{ return Packet2dd(_mm256_sub_pd(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet2dd pnegate(const Packet2dd& a) { return Packet2dd(pnegate(a.v)); }
template<> EIGEN_STRONG_INLINE Packet2dd pdconj(const Packet2dd& a)
{
  const __m256d mask = _mm256_castsi256_pd(_mm256_set_epi32(0x80000000,0x0,0x0,0x0,0x80000000,0x0,0x0,0x0));
  return Packet2dd(_mm256_xor_pd(a.v,mask));
}
template<> EIGEN_STRONG_INLINE Packet2dd pconj(const Packet2dd& a) { return a; }

template<> EIGEN_STRONG_INLINE Packet2dd pmul<Packet2dd>(const Packet2dd& a, const Packet2dd& b)
{
#if 0
  const __m256d mask = _mm256_castsi256_pd(_mm256_set_epi32(0xffffffff,0xffffffff,0x0,0x0,
                                                            0xffffffff,0xffffffff,0x0,0x0));
  return Packet2dd(_mm256_add_pd(_mm256_and_pd(mask,
                                               _mm256_mul_pd(a.v, _mm256_shuffle_pd(b.v,b.v,0x0))),
                                 _mm256_mul_pd(b.v, _mm256_shuffle_pd(a.v,a.v,0x0))));
#else
#ifdef __FMA__ // not sure if this is actually faster.
  return Packet2dd(_mm256_fmadd_pd(b.v,
                                   _mm256_movedup_pd(a.v),
                                   _mm256_blend_pd(_mm256_setzero_pd(),
                                                   _mm256_mul_pd(a.v, _mm256_movedup_pd(b.v)),
                                                   0xa)));
#else
  return Packet2dd(_mm256_add_pd(_mm256_mul_pd(b.v,_mm256_movedup_pd(a.v)),
                                 _mm256_blend_pd(_mm256_setzero_pd(),
                                                 _mm256_mul_pd(a.v, _mm256_movedup_pd(b.v)),
                                                 0xa)));
#endif
#endif
}

template<> EIGEN_STRONG_INLINE Packet2dd pand   <Packet2dd>(const Packet2dd& a, const Packet2dd& b)
{ return Packet2dd(_mm256_and_pd(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet2dd por    <Packet2dd>(const Packet2dd& a, const Packet2dd& b)
{ return Packet2dd(_mm256_or_pd(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet2dd pxor   <Packet2dd>(const Packet2dd& a, const Packet2dd& b)
{ return Packet2dd(_mm256_xor_pd(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet2dd pandnot<Packet2dd>(const Packet2dd& a, const Packet2dd& b)
{ return Packet2dd(_mm256_andnot_pd(a.v,b.v)); }

template<> EIGEN_STRONG_INLINE Packet2dd pload <Packet2dd>(const duals::dual<double>* from)
{ EIGEN_DEBUG_ALIGNED_LOAD return Packet2dd(pload<Packet4d>((const double*)from)); }
template<> EIGEN_STRONG_INLINE Packet2dd ploadu<Packet2dd>(const duals::dual<double>* from)
{ EIGEN_DEBUG_UNALIGNED_LOAD return Packet2dd(ploadu<Packet4d>((const double*)from)); }

template<> EIGEN_STRONG_INLINE Packet2dd pset1<Packet2dd>(const duals::dual<double>& from)
{
  // in case casting to a __m128d* is really not safe, then we can still fallback to this version: (much slower though)
//   return Packet2dd(_mm256_loadu2_m128d((const double*)&from,(const double*)&from));
    return Packet2dd(_mm256_broadcast_pd((const __m128d*)(const void*)&from));
}

template<> EIGEN_STRONG_INLINE Packet2dd ploaddup<Packet2dd>(const duals::dual<double>* from)
{ return pset1<Packet2dd>(*from); }

template<> EIGEN_STRONG_INLINE void pstore <duals::dual<double> >(duals::dual<double> *   to, const Packet2dd& from)
{ EIGEN_DEBUG_ALIGNED_STORE pstore((double*)to, from.v); }
template<> EIGEN_STRONG_INLINE void pstoreu<duals::dual<double> >(duals::dual<double> *   to, const Packet2dd& from)
{ EIGEN_DEBUG_UNALIGNED_STORE pstoreu((double*)to, from.v); }

template<> EIGEN_DEVICE_FUNC inline Packet2dd pgather<duals::dual<double>, Packet2dd>(const duals::dual<double>* from,
                                                                                      Index stride)
{
  return Packet2dd(_mm256_set_pd(duals::dpart(from[1*stride]), duals::rpart(from[1*stride]),
                                 duals::dpart(from[0*stride]), duals::rpart(from[0*stride])));
}

template<> EIGEN_DEVICE_FUNC inline void pscatter<duals::dual<double>, Packet2dd>(duals::dual<double>* to,
                                                                                  const Packet2dd& from, Index stride)
{
  __m128d low = _mm256_extractf128_pd(from.v, 0);
  to[stride*0] = duals::dual<double>(_mm_cvtsd_f64(low), _mm_cvtsd_f64(_mm_shuffle_pd(low, low, 1)));
  __m128d high = _mm256_extractf128_pd(from.v, 1);
  to[stride*1] = duals::dual<double>(_mm_cvtsd_f64(high), _mm_cvtsd_f64(_mm_shuffle_pd(high, high, 1)));
}

template<> EIGEN_STRONG_INLINE duals::dual<double> pfirst<Packet2dd>(const Packet2dd& a)
{
  __m128d low = _mm256_extractf128_pd(a.v, 0);
  EIGEN_ALIGN16 double res[2];
  _mm_store_pd(res, low);
  return duals::dual<double>(res[0],res[1]);
}

template<> EIGEN_STRONG_INLINE Packet2dd preverse(const Packet2dd& a) {
  __m256d result = _mm256_permute2f128_pd(a.v, a.v, 1);
  return Packet2dd(result);
}

template<> EIGEN_STRONG_INLINE duals::dual<double> predux<Packet2dd>(const Packet2dd& a)
{
  return predux(padd(Packet1dd(_mm256_extractf128_pd(a.v,0)),
                     Packet1dd(_mm256_extractf128_pd(a.v,1))));
}

template<> EIGEN_STRONG_INLINE Packet2dd preduxp<Packet2dd>(const Packet2dd* vecs)
{
  Packet4d t0 = _mm256_permute2f128_pd(vecs[0].v,vecs[1].v, 0 + (2<<4));
  Packet4d t1 = _mm256_permute2f128_pd(vecs[0].v,vecs[1].v, 1 + (3<<4));

  return Packet2dd(_mm256_add_pd(t0,t1));
}

template<> EIGEN_STRONG_INLINE duals::dual<double> predux_mul<Packet2dd>(const Packet2dd& a)
{
  return pfirst(pmul(Packet1dd(_mm256_extractf128_pd(a.v,0)),
                     Packet1dd(_mm256_extractf128_pd(a.v,1))));
}

template<int Offset>
struct palign_impl<Offset,Packet2dd>
{
  static EIGEN_STRONG_INLINE void run(Packet2dd& first, const Packet2dd& second)
  {
    if (Offset==0) return;
    palign_impl<Offset*2,Packet4d>::run(first.v, second.v);
  }
};

#if 0 // TODO
template<> struct dconj_helper<Packet2dd, Packet2dd, false,true>
{
  EIGEN_STRONG_INLINE Packet2dd pmadd(const Packet2dd& x, const Packet2dd& y, const Packet2dd& c) const
  { return padd(pmul(x,y),c); }
  EIGEN_STRONG_INLINE Packet2dd pmul(const Packet2dd& a, const Packet2dd& b) const
  { return internal::pmul(a, pdconj(b)); }
};

template<> struct dconj_helper<Packet2dd, Packet2dd, true,false>
{
  EIGEN_STRONG_INLINE Packet2dd pmadd(const Packet2dd& x, const Packet2dd& y, const Packet2dd& c) const
  { return padd(pmul(x,y),c); }
  EIGEN_STRONG_INLINE Packet2dd pmul(const Packet2dd& a, const Packet2dd& b) const
  { return internal::pmul(pdconj(a), b); }
};

template<> struct dconj_helper<Packet2dd, Packet2dd, true,true>
{
  EIGEN_STRONG_INLINE Packet2dd pmadd(const Packet2dd& x, const Packet2dd& y, const Packet2dd& c) const
  { return padd(pmul(x,y),c); }
  EIGEN_STRONG_INLINE Packet2dd pmul(const Packet2dd& a, const Packet2dd& b) const
  { return pdconj(internal::pmul(a, b)); }
};
EIGEN_MAKE_DCONJ_HELPER_DUAL_REAL(Packet2dd,Packet4d)
#endif


template<> EIGEN_STRONG_INLINE Packet2dd pdiv<Packet2dd>(const Packet2dd& a, const Packet2dd& b)
{
#if 1
  // help gcc to not use 12 registers :P
  __m256d y1 = _mm256_setzero_pd();
  __m256d y2 = _mm256_blend_pd(a.v, y1, 0xa);
  __m256d y4 = _mm256_movedup_pd(b.v);
  y4 = _mm256_mul_pd(a.v, y4);
  __m256d y0 = _mm256_movedup_pd(a.v);
  y0 = _mm256_mul_pd(b.v, y0);
  y0 = _mm256_sub_pd(y4, y0);
  y0 = _mm256_blend_pd(y1, y0, 0xa);
  y0 = _mm256_add_pd(y2, y0);
  y2 = _mm256_blend_pd(b.v, y1, 0xa);
  __m256d y3 = _mm256_mul_pd(b.v, b.v);
  y1 = _mm256_unpacklo_pd(y1, y3);
  y1 = _mm256_add_pd(y2, y1);
  return Packet2dd(_mm256_div_pd(y0, y1));
#else

  const __m256d mask = _mm256_castsi256_pd(_mm256_setr_epi32(0xffffffff,0xffffffff,0x00000000,0x00000000,
                                                             0xffffffff,0xffffffff,0x00000000,0x00000000));
  return Packet2dd(_mm256_div_pd
                   (_mm256_add_pd(_mm256_and_pd(mask, a.v),
                                  _mm256_andnot_pd(mask,
                                                   _mm256_sub_pd(_mm256_mul_pd(a.v,
                                                                               _mm256_shuffle_pd(b.v,b.v,0x0)),
                                                                 _mm256_mul_pd(b.v,
                                                                               _mm256_shuffle_pd(a.v,a.v,0x0))))),
                    _mm256_add_pd(_mm256_and_pd(mask, b.v),
                                  _mm256_andnot_pd(mask,
                                                   _mm256_mul_pd(_mm256_shuffle_pd(b.v,b.v,0x0),
                                                                 _mm256_shuffle_pd(b.v,b.v,0x0))))));
#endif
}

EIGEN_DEVICE_FUNC inline void
ptranspose(PacketBlock<Packet4df,4>& kernel) {
  __m256d P0 = _mm256_castps_pd(kernel.packet[0].v);
  __m256d P1 = _mm256_castps_pd(kernel.packet[1].v);
  __m256d P2 = _mm256_castps_pd(kernel.packet[2].v);
  __m256d P3 = _mm256_castps_pd(kernel.packet[3].v);
  __m256d T0 = _mm256_shuffle_pd(P0, P1, 15);
  __m256d T1 = _mm256_shuffle_pd(P0, P1, 0);
  __m256d T2 = _mm256_shuffle_pd(P2, P3, 15);
  __m256d T3 = _mm256_shuffle_pd(P2, P3, 0);
  kernel.packet[1].v = _mm256_castpd_ps(_mm256_permute2f128_pd(T0, T2, 32));
  kernel.packet[3].v = _mm256_castpd_ps(_mm256_permute2f128_pd(T0, T2, 49));
  kernel.packet[0].v = _mm256_castpd_ps(_mm256_permute2f128_pd(T1, T3, 32));
  kernel.packet[2].v = _mm256_castpd_ps(_mm256_permute2f128_pd(T1, T3, 49));
}

EIGEN_DEVICE_FUNC inline void
ptranspose(PacketBlock<Packet2dd,2>& kernel) {
  __m256d tmp = _mm256_permute2f128_pd(kernel.packet[0].v, kernel.packet[1].v, 0+(2<<4));
  kernel.packet[1].v = _mm256_permute2f128_pd(kernel.packet[0].v, kernel.packet[1].v, 1+(3<<4));
  kernel.packet[0].v = tmp;
}

template<> EIGEN_STRONG_INLINE Packet4df pinsertfirst(const Packet4df& a, duals::dual<float> b)
{
  return Packet4df(_mm256_blend_ps(a.v,pset1<Packet4df>(b).v,1|2));
}

template<> EIGEN_STRONG_INLINE Packet2dd pinsertfirst(const Packet2dd& a, duals::dual<double> b)
{
  return Packet2dd(_mm256_blend_pd(a.v,pset1<Packet2dd>(b).v,1|2));
}

template<> EIGEN_STRONG_INLINE Packet4df pinsertlast(const Packet4df& a, duals::dual<float> b)
{
  return Packet4df(_mm256_blend_ps(a.v,pset1<Packet4df>(b).v,(1<<7)|(1<<6)));
}

template<> EIGEN_STRONG_INLINE Packet2dd pinsertlast(const Packet2dd& a, duals::dual<double> b)
{
  return Packet2dd(_mm256_blend_pd(a.v,pset1<Packet2dd>(b).v,(1<<3)|(1<<2)));
}

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_DUAL_AVX_H

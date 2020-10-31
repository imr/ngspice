// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2019 Michael Tesch <tesch1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_DUAL_SSE_H
#define EIGEN_DUAL_SSE_H

namespace Eigen {

namespace internal {

//---------- float ----------
struct Packet2df
{
  EIGEN_STRONG_INLINE Packet2df() {}
  EIGEN_STRONG_INLINE explicit Packet2df(const __m128& a) : v(a) {}
  __m128  v;
};

// Use the packet_traits defined in AVX/Dual.h instead if we're going
// to leverage AVX instructions.
#ifndef EIGEN_VECTORIZE_AVX
template<> struct packet_traits<duals::dual<float> >  : default_packet_traits
{
  typedef Packet2df type;
  typedef Packet2df half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 2,
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
    HasSetLinear = 0,
    HasBlend = 1
  };
};
#endif

template<> struct unpacket_traits<Packet2df> {
  typedef duals::dual<float> type;
  enum {size=2, alignment=Aligned16, masked_load_available=false, masked_store_available=false, vectorizable=true};
  typedef Packet2df half;
};

template<> EIGEN_STRONG_INLINE Packet2df padd<Packet2df>(const Packet2df& a, const Packet2df& b)
{ return Packet2df(_mm_add_ps(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet2df psub<Packet2df>(const Packet2df& a, const Packet2df& b)
{ return Packet2df(_mm_sub_ps(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet2df pnegate(const Packet2df& a)
{
  const __m128 mask = _mm_castsi128_ps(_mm_setr_epi32(0x80000000,0x80000000,0x80000000,0x80000000));
  return Packet2df(_mm_xor_ps(a.v,mask));
}
template<> EIGEN_STRONG_INLINE Packet2df pdconj(const Packet2df& a)
{
  const __m128 mask = _mm_castsi128_ps(_mm_setr_epi32(0x00000000,0x80000000,0x00000000,0x80000000));
  return Packet2df(_mm_xor_ps(a.v,mask));
}
template<> EIGEN_STRONG_INLINE Packet2df pconj(const Packet2df& a) { return a; }

template<> EIGEN_STRONG_INLINE Packet2df pmul<Packet2df>(const Packet2df& a, const Packet2df& b)
{
#if defined(EIGEN_VECTORIZE_SSE3)
  const __m128 mask = _mm_castsi128_ps(_mm_setr_epi32(0x00000000,0xffffffff,0x00000000,0xffffffff));
  return Packet2df(_mm_add_ps(_mm_and_ps(mask, _mm_mul_ps(a.v, _mm_moveldup_ps(b.v))),
                              _mm_mul_ps(b.v, _mm_moveldup_ps(a.v))));
#else
  //TODO-use avx instructions instead?
  const __m128 mask = _mm_castsi128_ps(_mm_setr_epi32(0x00000000,0xffffffff,0x00000000,0xffffffff));
  return Packet2df(_mm_add_ps(_mm_and_ps(mask, _mm_mul_ps(a.v, _mm_moveldup_ps(b.v))),
                              _mm_mul_ps(b.v, _mm_moveldup_ps(a.v))));
#endif
}

template<> EIGEN_STRONG_INLINE Packet2df pand   <Packet2df>(const Packet2df& a, const Packet2df& b)
{ return Packet2df(_mm_and_ps(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet2df por    <Packet2df>(const Packet2df& a, const Packet2df& b)
{ return Packet2df(_mm_or_ps(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet2df pxor   <Packet2df>(const Packet2df& a, const Packet2df& b)
{ return Packet2df(_mm_xor_ps(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet2df pandnot<Packet2df>(const Packet2df& a, const Packet2df& b)
{ return Packet2df(_mm_andnot_ps(a.v,b.v)); }

template<> EIGEN_STRONG_INLINE Packet2df pload <Packet2df>(const duals::dual<float>* from)
{ EIGEN_DEBUG_ALIGNED_LOAD return Packet2df(pload<Packet4f>(&numext::real_ref(*from))); }
template<> EIGEN_STRONG_INLINE Packet2df ploadu<Packet2df>(const duals::dual<float>* from)
{ EIGEN_DEBUG_UNALIGNED_LOAD return Packet2df(ploadu<Packet4f>(&numext::real_ref(*from))); }

template<> EIGEN_STRONG_INLINE Packet2df pset1<Packet2df>(const duals::dual<float>&  from)
{
  Packet2df res;
#if EIGEN_GNUC_AT_MOST(4,2)
  // Workaround annoying "may be used uninitialized in this function" warning with gcc 4.2
  res.v = _mm_loadl_pi(_mm_set1_ps(0.0f), reinterpret_cast<const __m64*>(&from));
#elif EIGEN_GNUC_AT_LEAST(4,6)
  // Suppress annoying "may be used uninitialized in this function" warning with gcc >= 4.6
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wuninitialized"
  #pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
  res.v = _mm_loadl_pi(res.v, (const __m64*)&from);
  #pragma GCC diagnostic pop
#else
  res.v = _mm_loadl_pi(res.v, (const __m64*)&from);
#endif
  return Packet2df(_mm_movelh_ps(res.v,res.v));
}

template<> EIGEN_STRONG_INLINE Packet2df ploaddup<Packet2df>(const duals::dual<float>* from)
{ return pset1<Packet2df>(*from); }

template<> EIGEN_STRONG_INLINE void pstore <duals::dual<float> >(duals::dual<float> *   to, const Packet2df& from)
{ EIGEN_DEBUG_ALIGNED_STORE pstore(&numext::real_ref(*to), Packet4f(from.v)); }
template<> EIGEN_STRONG_INLINE void pstoreu<duals::dual<float> >(duals::dual<float> *   to, const Packet2df& from)
{ EIGEN_DEBUG_UNALIGNED_STORE pstoreu(&numext::real_ref(*to), Packet4f(from.v)); }


template<> EIGEN_DEVICE_FUNC inline Packet2df pgather<duals::dual<float>, Packet2df>(const duals::dual<float>* from,
                                                                                     Index stride)
{
  return Packet2df(_mm_set_ps(duals::dpart(from[1*stride]), duals::rpart(from[1*stride]),
                              duals::dpart(from[0*stride]), duals::rpart(from[0*stride])));
}

template<> EIGEN_DEVICE_FUNC inline void pscatter<duals::dual<float>, Packet2df>(duals::dual<float>* to,
                                                                                 const Packet2df& from, Index stride)
{
  to[stride*0] = duals::dual<float>(_mm_cvtss_f32(_mm_shuffle_ps(from.v, from.v, 0)),
                                    _mm_cvtss_f32(_mm_shuffle_ps(from.v, from.v, 1)));
  to[stride*1] = duals::dual<float>(_mm_cvtss_f32(_mm_shuffle_ps(from.v, from.v, 2)),
                                    _mm_cvtss_f32(_mm_shuffle_ps(from.v, from.v, 3)));
}

template<> EIGEN_STRONG_INLINE void prefetch<duals::dual<float> >(const duals::dual<float> *   addr)
{ _mm_prefetch((SsePrefetchPtrType)(addr), _MM_HINT_T0); }

template<> EIGEN_STRONG_INLINE duals::dual<float>  pfirst<Packet2df>(const Packet2df& a)
{
  #if EIGEN_GNUC_AT_MOST(4,3)
  // Workaround gcc 4.2 ICE - this is not performance wise ideal, but who cares...
  // This workaround also fix invalid code generation with gcc 4.3
  EIGEN_ALIGN16 duals::dual<float> res[2];
  _mm_store_ps((float*)res, a.v);
  return res[0];
  #else
  duals::dual<float> res;
  _mm_storel_pi((__m64*)&res, a.v);
  return res;
  #endif
}

template<> EIGEN_STRONG_INLINE Packet2df preverse(const Packet2df& a)
{ return Packet2df(_mm_castpd_ps(preverse(Packet2d(_mm_castps_pd(a.v))))); }

template<> EIGEN_STRONG_INLINE duals::dual<float> predux<Packet2df>(const Packet2df& a)
{
  return pfirst(Packet2df(_mm_add_ps(a.v, _mm_movehl_ps(a.v,a.v))));
}

template<> EIGEN_STRONG_INLINE Packet2df preduxp<Packet2df>(const Packet2df* vecs)
{
  return Packet2df(_mm_add_ps(_mm_movelh_ps(vecs[0].v,vecs[1].v),
                              _mm_movehl_ps(vecs[1].v,vecs[0].v)));
}

template<> EIGEN_STRONG_INLINE duals::dual<float> predux_mul<Packet2df>(const Packet2df& a)
{
  return pfirst(pmul(a, Packet2df(_mm_movehl_ps(a.v,a.v))));
}

template<int Offset>
struct palign_impl<Offset,Packet2df>
{
  static EIGEN_STRONG_INLINE void run(Packet2df& first, const Packet2df& second)
  {
    if (Offset==1)
    {
      first.v = _mm_movehl_ps(first.v, first.v);
      first.v = _mm_movelh_ps(first.v, second.v);
    }
  }
};

#if 0 // TODO
template<> struct dconj_helper<Packet2df, Packet2df, false,true>
{
  EIGEN_STRONG_INLINE Packet2df pmadd(const Packet2df& x, const Packet2df& y, const Packet2df& c) const
  { return padd(pmul(x,y),c); }
  EIGEN_STRONG_INLINE Packet2df pmul(const Packet2df& a, const Packet2df& b) const
  { return internal::pmul(a, pdconj(b)); }
};

template<> struct dconj_helper<Packet2df, Packet2df, true,false>
{
  EIGEN_STRONG_INLINE Packet2df pmadd(const Packet2df& x, const Packet2df& y, const Packet2df& c) const
  { return padd(pmul(x,y),c); }
  EIGEN_STRONG_INLINE Packet2df pmul(const Packet2df& a, const Packet2df& b) const
  { return internal::pmul(pdconj(a), b); }
};

template<> struct dconj_helper<Packet2df, Packet2df, true,true>
{
  EIGEN_STRONG_INLINE Packet2df pmadd(const Packet2df& x, const Packet2df& y, const Packet2df& c) const
  { return padd(pmul(x,y),c); }
  EIGEN_STRONG_INLINE Packet2df pmul(const Packet2df& a, const Packet2df& b) const
  { return pdconj(internal::pmul(a, b)); }
};
EIGEN_MAKE_DCONJ_HELPER_DUAL_REAL(Packet2df,Packet4f)
#endif

template<> EIGEN_STRONG_INLINE Packet2df pdiv<Packet2df>(const Packet2df& a, const Packet2df& b)
{
  // TODO optimize it for SSE3 and 4
  const __m128 mask = _mm_castsi128_ps(_mm_setr_epi32(0xffffffff,0x00000000,0xffffffff,0x00000000));
  return Packet2df(_mm_div_ps(_mm_add_ps(_mm_and_ps(mask, a.v),
                                         _mm_andnot_ps(mask,
                                                       _mm_sub_ps(_mm_mul_ps(a.v, vec4f_swizzle1(b.v, 0, 0, 2, 2)),
                                                                  _mm_mul_ps(b.v, vec4f_swizzle1(a.v, 0, 0, 2, 2))))),
                              _mm_add_ps(_mm_and_ps(mask, b.v),
                                         _mm_andnot_ps(mask,
                                                       _mm_mul_ps(vec4f_swizzle1(b.v, 0, 0, 2, 2),
                                                                  vec4f_swizzle1(b.v, 0, 0, 2, 2))))));
}

//---------- double ----------
struct Packet1dd
{
  EIGEN_STRONG_INLINE Packet1dd() {}
  EIGEN_STRONG_INLINE explicit Packet1dd(const __m128d& a) : v(a) {}
  __m128d  v;
};

// Use the packet_traits defined in AVX/PacketMath.h instead if we're going
// to leverage AVX instructions.
#ifndef EIGEN_VECTORIZE_AVX
template<> struct packet_traits<duals::dual<double> >  : default_packet_traits
{
  typedef Packet1dd type;
  typedef Packet1dd half;
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

template<> struct unpacket_traits<Packet1dd> {
  typedef duals::dual<double> type;
  enum {size=1, alignment=Aligned16, masked_load_available=false, masked_store_available=false, vectorizable=true};
  typedef Packet1dd half;
};

template<> EIGEN_STRONG_INLINE Packet1dd padd<Packet1dd>(const Packet1dd& a, const Packet1dd& b)
{ return Packet1dd(_mm_add_pd(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet1dd psub<Packet1dd>(const Packet1dd& a, const Packet1dd& b)
{ return Packet1dd(_mm_sub_pd(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet1dd pnegate(const Packet1dd& a) { return Packet1dd(pnegate(Packet2d(a.v))); }
template<> EIGEN_STRONG_INLINE Packet1dd pdconj(const Packet1dd& a)
{
  const __m128d mask = _mm_castsi128_pd(_mm_set_epi32(0x80000000,0x0,0x0,0x0));
  return Packet1dd(_mm_xor_pd(a.v,mask));
}
template<> EIGEN_STRONG_INLINE Packet1dd pconj(const Packet1dd& a) { return a; }

template<> EIGEN_STRONG_INLINE Packet1dd pmul<Packet1dd>(const Packet1dd& a, const Packet1dd& b)
{
  //#ifdef EIGEN_VECTORIZE_SSE3
  //TODO
  const __m128d mask = _mm_castsi128_pd(_mm_set_epi32(0xffffffff,0xffffffff,0x00000000,0x00000000));
  return Packet1dd(_mm_add_pd(_mm_and_pd(mask,
                                         _mm_mul_pd(vec2d_swizzle1(a.v, 0, 1),
                                                    vec2d_swizzle1(b.v, 0, 0))),
                              _mm_mul_pd(vec2d_swizzle1(a.v, 0, 0),
                                         vec2d_swizzle1(b.v, 0, 1))));
  //#else
  //#endif
}

template<> EIGEN_STRONG_INLINE Packet1dd pand   <Packet1dd>(const Packet1dd& a, const Packet1dd& b)
{ return Packet1dd(_mm_and_pd(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet1dd por    <Packet1dd>(const Packet1dd& a, const Packet1dd& b)
{ return Packet1dd(_mm_or_pd(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet1dd pxor   <Packet1dd>(const Packet1dd& a, const Packet1dd& b)
{ return Packet1dd(_mm_xor_pd(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet1dd pandnot<Packet1dd>(const Packet1dd& a, const Packet1dd& b)
{ return Packet1dd(_mm_andnot_pd(a.v,b.v)); }

// FIXME force unaligned load, this is a temporary fix
template<> EIGEN_STRONG_INLINE Packet1dd pload <Packet1dd>(const duals::dual<double>* from)
{ EIGEN_DEBUG_ALIGNED_LOAD return Packet1dd(pload<Packet2d>((const double*)from)); }
template<> EIGEN_STRONG_INLINE Packet1dd ploadu<Packet1dd>(const duals::dual<double>* from)
{ EIGEN_DEBUG_UNALIGNED_LOAD return Packet1dd(ploadu<Packet2d>((const double*)from)); }
template<> EIGEN_STRONG_INLINE Packet1dd pset1<Packet1dd>(const duals::dual<double>&  from)
{ /* here we really have to use unaligned loads :( */ return ploadu<Packet1dd>(&from); }

template<> EIGEN_STRONG_INLINE Packet1dd ploaddup<Packet1dd>(const duals::dual<double>* from)
{ return pset1<Packet1dd>(*from); }

// FIXME force unaligned store, this is a temporary fix
template<> EIGEN_STRONG_INLINE void pstore <duals::dual<double> >(duals::dual<double> *   to, const Packet1dd& from)
{ EIGEN_DEBUG_ALIGNED_STORE pstore((double*)to, Packet2d(from.v)); }
template<> EIGEN_STRONG_INLINE void pstoreu<duals::dual<double> >(duals::dual<double> *   to, const Packet1dd& from)
{ EIGEN_DEBUG_UNALIGNED_STORE pstoreu((double*)to, Packet2d(from.v)); }

template<> EIGEN_STRONG_INLINE void prefetch<duals::dual<double> >(const duals::dual<double> *   addr)
{ _mm_prefetch((SsePrefetchPtrType)(addr), _MM_HINT_T0); }

template<> EIGEN_STRONG_INLINE duals::dual<double>  pfirst<Packet1dd>(const Packet1dd& a)
{
  EIGEN_ALIGN16 double res[2];
  _mm_store_pd(res, a.v);
  return duals::dual<double>(res[0],res[1]);
}

template<> EIGEN_STRONG_INLINE Packet1dd preverse(const Packet1dd& a) { return a; }

template<> EIGEN_STRONG_INLINE duals::dual<double> predux<Packet1dd>(const Packet1dd& a)
{
  return pfirst(a);
}

template<> EIGEN_STRONG_INLINE Packet1dd preduxp<Packet1dd>(const Packet1dd* vecs)
{
  return vecs[0];
}

template<> EIGEN_STRONG_INLINE duals::dual<double> predux_mul<Packet1dd>(const Packet1dd& a)
{
  return pfirst(a);
}

template<int Offset>
struct palign_impl<Offset,Packet1dd>
{
  static EIGEN_STRONG_INLINE void run(Packet1dd& /*first*/, const Packet1dd& /*second*/)
  {
    // FIXME is it sure we never have to align a Packet1dd?
    // Even though a duals::dual<double> has 16 bytes, it is not necessarily aligned on a 16 bytes boundary...
  }
};

#if 0 // TODO
template<> struct dconj_helper<Packet1dd, Packet1dd, false,true>
{
  EIGEN_STRONG_INLINE Packet1dd pmadd(const Packet1dd& x, const Packet1dd& y, const Packet1dd& c) const
  { return padd(pmul(x,y),c); }
  EIGEN_STRONG_INLINE Packet1dd pmul(const Packet1dd& a, const Packet1dd& b) const
  { return internal::pmul(a, pdconj(b)); }
};

template<> struct dconj_helper<Packet1dd, Packet1dd, true,false>
{
  EIGEN_STRONG_INLINE Packet1dd pmadd(const Packet1dd& x, const Packet1dd& y, const Packet1dd& c) const
  { return padd(pmul(x,y),c); }
  EIGEN_STRONG_INLINE Packet1dd pmul(const Packet1dd& a, const Packet1dd& b) const
  { return internal::pmul(pdconj(a), b); }
};

template<> struct dconj_helper<Packet1dd, Packet1dd, true,true>
{
  EIGEN_STRONG_INLINE Packet1dd pmadd(const Packet1dd& x, const Packet1dd& y, const Packet1dd& c) const
  { return padd(pmul(x,y),c); }
  EIGEN_STRONG_INLINE Packet1dd pmul(const Packet1dd& a, const Packet1dd& b) const
  { return pdconj(internal::pmul(a, b)); }
};
EIGEN_MAKE_DCONJ_HELPER_DUAL_REAL(Packet1dd,Packet2d)
#endif

template<> EIGEN_STRONG_INLINE Packet1dd pdiv<Packet1dd>(const Packet1dd& a, const Packet1dd& b)
{
  // TODO optimize it for SSE3 and 4
  //const __m128d mask = _mm_castsi128_pd(_mm_set_epi32(0x0,0x0,0x80000000,0x0));
  //return Packet1cd(_mm_add_pd(_mm_mul_pd(vec2d_swizzle1(a.v, 0, 0), b.v),
  //                            _mm_xor_pd(_mm_mul_pd(vec2d_swizzle1(a.v, 1, 1),
  //                                                  vec2d_swizzle1(b.v, 1, 0)), mask)));
  // TODO optimize it for SSE3 and 4
  const __m128d mask = _mm_castsi128_pd(_mm_setr_epi32(0xffffffff,0xffffffff,0x00000000,0x00000000));
  return Packet1dd(_mm_div_pd(_mm_add_pd(_mm_and_pd(mask, a.v),
                                         _mm_andnot_pd(mask,
                                                       _mm_sub_pd(_mm_mul_pd(vec2d_swizzle1(a.v, 0, 1),
                                                                             vec2d_swizzle1(b.v, 0, 0)),
                                                                  _mm_mul_pd(vec2d_swizzle1(a.v, 0, 0),
                                                                             vec2d_swizzle1(b.v, 0, 1))))),
                              _mm_add_pd(_mm_and_pd(mask, b.v),
                                         _mm_andnot_pd(mask,
                                                       _mm_mul_pd(vec2d_swizzle1(b.v, 0, 0),
                                                                  vec2d_swizzle1(b.v, 0, 0))))));
}

EIGEN_DEVICE_FUNC inline void
ptranspose(PacketBlock<Packet2df,2>& kernel) {
  __m128d w1 = _mm_castps_pd(kernel.packet[0].v);
  __m128d w2 = _mm_castps_pd(kernel.packet[1].v);

  __m128 tmp = _mm_castpd_ps(_mm_unpackhi_pd(w1, w2));
  kernel.packet[0].v = _mm_castpd_ps(_mm_unpacklo_pd(w1, w2));
  kernel.packet[1].v = tmp;
}

template<>  EIGEN_STRONG_INLINE Packet2df pblend(const Selector<2>& ifPacket,
                                                 const Packet2df& thenPacket,
                                                 const Packet2df& elsePacket)
{
  __m128d result = pblend<Packet2d>(ifPacket, _mm_castps_pd(thenPacket.v), _mm_castps_pd(elsePacket.v));
  return Packet2df(_mm_castpd_ps(result));
}

template<> EIGEN_STRONG_INLINE Packet2df pinsertfirst(const Packet2df& a, duals::dual<float> b)
{
  return Packet2df(_mm_loadl_pi(a.v, reinterpret_cast<const __m64*>(&b)));
}

template<> EIGEN_STRONG_INLINE Packet1dd pinsertfirst(const Packet1dd&, duals::dual<double> b)
{
  return pset1<Packet1dd>(b);
}

template<> EIGEN_STRONG_INLINE Packet2df pinsertlast(const Packet2df& a, duals::dual<float> b)
{
  return Packet2df(_mm_loadh_pi(a.v, reinterpret_cast<const __m64*>(&b)));
}

template<> EIGEN_STRONG_INLINE Packet1dd pinsertlast(const Packet1dd&, duals::dual<double> b)
{
  return pset1<Packet1dd>(b);
}

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_COMPLEX_SSE_H

/*******************************************************************************
 * Copyright 2020 Florian Ballenegger, Anamosic Ballenegger Design
 *******************************************************************************
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 ******************************************************************************/

#if USEX86INTRINSICS==1
#include <x86intrin.h>
#define vec2_MAX(a,b) _mm_max_pd(a,b)
#define vecN_MAX vec2_MAX
#define vec2_sqrt(a) _mm_sqrt_pd(a)
#define vecN_sqrt vec2_sqrt
static inline Vec2d vec2_blend(Vec2d fa, Vec2d tr, Vec2m mask)
{
	return _mm_blendv_pd(fa,tr, (Vec2d) mask);
}
#define vecN_blend vec2_blend
#else
#define vec2_blend vecN_blend
#endif

#ifdef USE_LIBSLEEF
#include <sleef.h>
#define vec2_exp(a) Sleef_expd2_u10(a)
#define vecN_exp vec2_exp
#define vec2_log(a) Sleef_logd2_u35(a)
#define vecN_log vec2_log
#ifndef USEX86INTRINSICS
#define vec2_MAX(a,b) Sleef_fmaxd2(a,b)
#define vecN_MAX vec2_MAX
#define vec2_sqrt(a) Sleef_sqrtd2_u35(a)
#define vecN_sqrt vec2_sqrt
#endif
#define vec2_fabs(a) Sleef_fabsd2(a)
#define vecN_fabs vec2_fabs
#define vec2_pow(a,b) Sleef_powd2_u10(a,vec2_SIMDTOVECTOR(b))
#define vecN_pow vec2_pow

#else

#ifdef HAS_LIBMVEC
Vec2d _ZGVbN2v_exp(Vec2d);
Vec2d _ZGVbN2v_log(Vec2d);
Vec2d _ZGVbN2vv_pow(Vec2d, Vec2d);

#define vec2_exp(a) _ZGVbN2v_exp(a)
#define vecN_exp vec2_exp
#define vec2_log(a) _ZGVbN2v_log(a)
#define vecN_log vec2_log
#define vec2_pow(a,b) _ZGVbN2vv_pow(a,b)
#define vecN_pow vec2_pow
#define vec2_fabs vecN_fabs

#endif /* HAS_LIBMVEC */
#endif /* not USE_LIBSLEEF */

#ifdef USE_SERIAL_FORM

#define vec2_SIMDTOVECTOR vecN_SIMDTOVECTOR
#define vec2_SIMDTOVECTORMASK vecN_SIMDTOVECTORMASK
#define vec2_StateAccess vecN_StateAccess
#define vec2_SIMDCOUNT vecN_SIMDCOUNT

#else

static inline Vec2d vec2_SIMDTOVECTOR(double val)
{
	return (Vec2d) {val,val};
}
static inline Vec2m vec2_SIMDTOVECTORMASK(int val)
{
	return (Vec2m) {val,val};
}
static inline Vec2d vec2_StateAccess(double* cktstate, Vec2m stateindexes)
{
	return (Vec2d) {
	 cktstate[stateindexes[0]],
	 cktstate[stateindexes[1]]
	};
}
static inline int vec2_SIMDCOUNT(Vec2m mask) {
	return (mask[0] ? 1 : 0) + (mask[1] ? 1 : 0) ;
}
#define vecN_SIMDTOVECTOR vec2_SIMDTOVECTOR
#define vecN_SIMDTOVECTORMASK vec2_SIMDTOVECTORMASK
#define vecN_StateAccess vec2_StateAccess
#define vecN_SIMDCOUNT vec2_SIMDCOUNT

#endif

#define vec2_StateStore vecN_StateStore
#define vec2_StateAdd vecN_StateAdd
#define vec2_StateSub vecN_StateSub


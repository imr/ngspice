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
#define vec4_MAX(a,b) _mm256_max_pd(a,b)
#define vecN_MAX vec4_MAX
#define vec4_sqrt(a) _mm256_sqrt_pd(a)
#define vecN_sqrt vec4_sqrt
static inline Vec4d vec4_blend(Vec4d fa, Vec4d tr, Vec4m mask)
{
	return _mm256_blendv_pd(fa,tr, (Vec4d) mask);
}
#define vecN_blend vec4_blend
#else
#define vec4_blend vecN_blend
#endif

#ifdef HAVE_LIBSLEEF
#include <sleef.h>
#define vec4_exp(a) Sleef_expd4_u10(a)
#define vecN_exp vec4_exp
#define vec4_log(a) Sleef_logd4_u35(a)
#define vecN_log vec4_log
#ifndef USEX86INTRINSICS
#define vec4_MAX(a,b) Sleef_fmaxd4(a,b)
#define vecN_MAX vec4_MAX
#define vec4_sqrt(a) Sleef_sqrtd4_u35(a)
#define vecN_sqrt vec4_sqrt
#endif
#define vec4_fabs(a) Sleef_fabsd4(a)
#define vecN_fabs vec4_fabs
#define vec4_pow(a,b) Sleef_powd4_u10(a,vec4_SIMDTOVECTOR(b))
#define vecN_pow vec4_pow

#else

#ifdef HAS_LIBMVEC
Vec4d _ZGVdN4v_exp(Vec4d);
Vec4d _ZGVdN4v_log(Vec4d);
Vec4d _ZGVdN4vv_pow(Vec4d, Vec4d);

#define vec4_exp(a) _ZGVdN4v_exp(a)
#define vecN_exp vec4_exp
#define vec4_log(a) _ZGVdN4v_log(a)
#define vecN_log vec4_log
#define vec4_pow(a,b) _ZGVdN4vv_pow(a,b)
#define vecN_pow vec4_pow
#define vec4_fabs vecN_fabs

#endif /* HAS_LIBMVEC */
#endif /* not HAVE_LIBSLEEF */

#ifdef USE_SERIAL_FORM

#define vec4_SIMDTOVECTOR vecN_SIMDTOVECTOR
#define vec4_SIMDTOVECTORMASK vecN_SIMDTOVECTORMASK
#define vec4_StateAccess vecN_StateAccess
#define vec4_SIMDCOUNT vecN_SIMDCOUNT

#else

static inline Vec4d vec4_SIMDTOVECTOR(double val)
{
	return (Vec4d) {val,val,val,val};
}
static inline Vec4m vec4_SIMDTOVECTORMASK(int val)
{
	return (Vec4m) {val,val,val,val};
}
static inline Vec4d vec4_StateAccess(double* cktstate, Vec4m stateindexes)
{
	return (Vec4d) {
	 cktstate[stateindexes[0]],
	 cktstate[stateindexes[1]],
	 cktstate[stateindexes[2]],
	 cktstate[stateindexes[3]]
	};
}
static inline int vec4_SIMDCOUNT(Vec4m mask) {
	return (mask[0] ? 1 : 0) + (mask[1] ? 1 : 0) + (mask[2] ? 1 : 0) + (mask[3] ? 1 : 0);
}
#define vecN_SIMDTOVECTOR vec4_SIMDTOVECTOR
#define vecN_SIMDTOVECTORMASK vec4_SIMDTOVECTORMASK
#define vecN_StateAccess vec4_StateAccess
#define vecN_SIMDCOUNT vec4_SIMDCOUNT

#endif

#define vec4_StateStore vecN_StateStore
#define vec4_StateAdd vecN_StateAdd
#define vec4_StateSub vecN_StateSub


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

#define vec8_blend vecN_blend

#ifdef USE_LIBSLEEF
#include <sleef.h>
#define vec8_exp(a) Sleef_expd8_u10(a)
#define vecN_exp vec8_exp
#define vec8_log(a) Sleef_logd8_u35(a)
#define vecN_log vec8_log
#ifndef USEX86INTRINSICS
#define vec8_MAX(a,b) Sleef_fmaxd8(a,b)
#define vecN_MAX vec8_MAX
#define vec8_sqrt(a) Sleef_sqrtd8_u35(a)
#define vecN_sqrt vec8_sqrt
#endif
#define vec8_fabs(a) Sleef_fabsd8(a)
#define vecN_fabs vec8_fabs
#define vec8_pow(a,b) Sleef_powd8_u10(a,vec8_SIMDTOVECTOR(b))
#define vecN_pow vec8_pow

#else

#define vec8_exp vecN_exp
#define vec8_log vecN_log
#define vec8_fabs vecN_fabs
#define vec8_pow vecN_pow
#ifndef USEX86INTRINSICS
#define vec8_MAX vecN_MAX
#define vec8_sqrt vecN_sqrt
#endif

#endif /* USE_LIBSLEEF */

#ifdef USE_SERIAL_FORM

#define vec8_SIMDTOVECTOR vecN_SIMDTOVECTOR
#define vec8_SIMDTOVECTORMASK vecN_SIMDTOVECTORMASK
#define vec8_StateAccess vecN_StateAccess
#define vec8_SIMDCOUNT vecN_SIMDCOUNT

#else

static inline Vec8d vec8_SIMDTOVECTOR(double val)
{
	return (Vec8d) {val,val,val,val,val,val,val,val};
}
static inline Vec8m vec8_SIMDTOVECTORMASK(int val)
{
	return (Vec8m) {val,val,val,val,val,val,val,val};
}
static inline Vec8d vec8_StateAccess(double* cktstate, Vec8m stateindexes)
{
	return (Vec8d) {
	 cktstate[stateindexes[0]],
	 cktstate[stateindexes[1]],
	 cktstate[stateindexes[2]],
	 cktstate[stateindexes[3]],
	 cktstate[stateindexes[4]],
	 cktstate[stateindexes[5]],
	 cktstate[stateindexes[6]],
	 cktstate[stateindexes[7]],
	};
}
static inline int vec8_SIMDCOUNT(Vec8m mask) {
	return (   mask[0] ? 1 : 0)
		+ (mask[1] ? 1 : 0)
		+ (mask[2] ? 1 : 0)
		+ (mask[3] ? 1 : 0)
		+ (mask[4] ? 1 : 0)
		+ (mask[5] ? 1 : 0)
		+ (mask[6] ? 1 : 0)
		+ (mask[7] ? 1 : 0);
}
#define vecN_SIMDTOVECTOR vec8_SIMDTOVECTOR
#define vecN_SIMDTOVECTORMASK vec8_SIMDTOVECTORMASK
#define vecN_StateAccess vec8_StateAccess
#define vecN_SIMDCOUNT vec8_SIMDCOUNT

#endif

#define vec8_StateStore vecN_StateStore
#define vec8_StateAdd vecN_StateAdd
#define vec8_StateSub vecN_StateSub


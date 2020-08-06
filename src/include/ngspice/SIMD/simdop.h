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

#ifndef NG_SIMD_OP_H
#define NG_SIMD_OP_H

#include "ngspice/SIMD/simdvector.h"

/* first include vector functions specialized for specific NSIMD */

#if NSIMD==4
#include "ngspice/SIMD/simdop4.h"
#endif /* NSIMD==4 */

#if NSIMD==8
#include "ngspice/SIMD/simdop8.h"
#endif /* NSIMD==8 */

#if NSIMD==2
#include "ngspice/SIMD/simdop2.h"
#endif /* NSIMD==2 */

/* now define missing vector functions in a generic manner */

inline VecNd vecN_broadcast(double x)
{
	VecNd res;
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(NSIMD)
	#endif
	for(int i=0;i<NSIMD;i++)
		res[i]=x;
	return res;
}

inline VecNd vecN_lu(double* array, VecNi indexes)
{
	VecNd res;
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(NSIMD)
	#endif
	for(int i=0;i<NSIMD;i++)
		res[i]=array[indexes[i]];
	return res;
}

#ifndef vecN_MAX
inline VecNd vecN_MAX(VecNd a, VecNd b)
{
	VecNd res;
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(NSIMD)
	#endif
	for(int i=0;i<NSIMD;i++)
		res[i]=(a[i] > b[i]) ? a[i] : b[i];
	return res;
}
#endif

#ifndef vecN_fabs
inline VecNd vecN_fabs(VecNd x)
{
	VecNd res;
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(NSIMD)
	#endif
	for(int i=0;i<NSIMD;i++)
		res[i]=fabs(x[i]);
	return res;
}
#endif

#ifndef vecN_sqrt
inline VecNd vecN_sqrt(VecNd x)
{
	VecNd res;
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(NSIMD)
	#endif
	for(int i=0;i<NSIMD;i++)
		res[i]=sqrt(x[i]);
	return res;
}
#endif

#ifndef vecN_pow
inline VecNd vecN_pow(VecNd x, double p)
{
	VecNd res;
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(NSIMD)
	#endif
	for(int i=0;i<NSIMD;i++)
		res[i]=log(x[i]);
	res = res*p;
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(NSIMD)
	#endif
	for(int i=0;i<NSIMD;i++)
		res[i]=exp(res[i]);
	return res;
}
#endif

#ifndef vecN_exp
inline VecNd vecN_exp(VecNd x)
{
	VecNd res;
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(NSIMD)
	#endif
	for(int i=0;i<NSIMD;i++)
		res[i]=exp(x[i]);
	return res;
}
#endif

#ifndef vecN_log
inline VecNd vecN_log(VecNd x)
{
	VecNd res;
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(NSIMD)
	#endif
	for(int i=0;i<NSIMD;i++)
		res[i]=log(x[i]);
	return res;
}
#endif

#ifndef vecN_blend
static inline VecNd vecN_blend(VecNd fa, VecNd tr, VecNm mask)
{
	VecNd r;
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(NSIMD)
	#endif
	for(int i=0;i<NSIMD;i++)
		r[i] = (mask[i]==0 ? fa[i] : tr[i]);
	return r;
}
#endif

#ifndef vecN_SIMDTOVECTOR
static inline VecNd vecN_SIMDTOVECTOR(double val)
{
	VecNd r;
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(NSIMD)
	#endif
	for(int i=0;i<NSIMD;i++)
		r[i] = val;
	return r;
}
#endif

#ifndef vecN_SIMDTOVECTORMASK
static inline VecNm vecN_SIMDTOVECTORMASK(int val)
{
	VecNm r;
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(NSIMD)
	#endif
	for(int i=0;i<NSIMD;i++)
		r[i] = val;
	return r;
}
#endif

#ifndef vecN_StateAccess
static inline VecNd vecN_StateAccess(double* cktstate, VecNm stateindexes)
{
	VecNd r;
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(NSIMD)
	#endif
	for(int i=0;i<NSIMD;i++)
		r[i] =  cktstate[stateindexes[i]];
	return r;
}
#endif

#ifndef vecN_SIMDCOUNT
static inline int vecN_SIMDCOUNT(VecNm mask) {
	int count=0;
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(NSIMD) reduction(+:count)
	#endif
	for(int i=0;i<NSIMD;i++)
	  if(mask[i])
	    count++;
	return count;
}
#endif

#ifndef vecN_StateStore
static inline void vecN_StateStore(double* cktstate, VecNm stateindexes, VecNd values)
{
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(NSIMD)
	#endif
	for(int idx=0;idx<NSIMD;idx++)
	{
		cktstate[stateindexes[idx]] = values[idx];
	}
}
#endif

#ifndef vecN_StateAdd
static inline void vecN_StateAdd(double* cktstate, VecNm stateindexes, VecNd values)
{
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(NSIMD)
	#endif
	for(int idx=0;idx<NSIMD;idx++)
	{
		cktstate[stateindexes[idx]] += values[idx];
	}
}
#endif

#ifndef vecN_StateSub
static inline void vecN_StateSub(double* cktstate, VecNm stateindexes, VecNd values)
{
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(NSIMD)
	#endif
	for(int idx=0;idx<NSIMD;idx++)
	{
		cktstate[stateindexes[idx]] -= values[idx];
	}
}
#endif

#endif /* NG_SIMD_OP_H */

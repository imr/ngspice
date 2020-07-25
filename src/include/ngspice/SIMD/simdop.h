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

inline VecNd vecN_broadcast(double x)
{
	VecNd res;
	for(int i=0;i<NSIMD;i++)
		res[i]=x;
	return res;
}

inline VecNd vecN_lu(double* array, VecNi indexes)
{
	VecNd res;
	for(int i=0;i<NSIMD;i++)
		res[i]=array[indexes[i]];
	return res;
}

inline VecNd vecN_MAX(VecNd a, VecNd b)
{
	VecNd res;
	for(int i=0;i<NSIMD;i++)
		res[i]=(a[i] > b[i]) ? a[i] : b[i];
	return res;
}

inline VecNd vecN_fabs(VecNd x)
{
	VecNd res;
	for(int i=0;i<NSIMD;i++)
		res[i]=fabs(x[i]);
	return res;
}

inline VecNd vecN_sqrt(VecNd x)
{
	VecNd res;
	for(int i=0;i<NSIMD;i++)
		res[i]=sqrt(x[i]);
	return res;
}

inline VecNd vecN_pow(VecNd x, double p)
{
	VecNd res;
	for(int i=0;i<NSIMD;i++)
		res[i]=log(x[i]);
	res = res*p;
	for(int i=0;i<NSIMD;i++)
		res[i]=exp(res[i]);
	return res;
}

inline VecNd vecN_exp(VecNd x)
{
	VecNd res;
	for(int i=0;i<NSIMD;i++)
		res[i]=exp(x[i]);
	return res;
}

inline VecNd vecN_log(VecNd x)
{
	VecNd res;
	for(int i=0;i<NSIMD;i++)
		res[i]=log(x[i]);
	return res;
}


#endif

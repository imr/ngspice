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

#include <math.h>

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3v32def.h"
#include "b3v32acm.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/devdefs.h"
#include "ngspice/suffix.h"

#include "ngspice/SIMD/simdvector.h"
#include "ngspice/SIMD/simdop.h"
#include "ngspice/SIMD/simdniinteg.h"

#define MAX_EXP 5.834617425e14
#define MIN_EXP 1.713908431e-15
#define EXP_THRESHOLD 34.0
#define EPSOX 3.453133e-11
#define EPSSI 1.03594e-10
#define Charge_q 1.60219e-19
#define DELTA_1 0.02
#define DELTA_2 0.02
#define DELTA_3 0.02
#define DELTA_4 0.02

#define SIMDANY(err) (err!=0)
#define SIMDIFYCMD(cmd) /* empty */
#define SIMDifySaveScope(sc) /* empty */

static inline VecNd vecN_SIMDLOADDATA(int idx, double data[7][NSIMD])
{
	VecNd r;
	for(int i=0;i<NSIMD;i++)
		r[i] =  data[idx][i];
	return r;
}


static inline int vecN_BSIM3v32_ACM_saturationCurrents
(
	BSIM3v32model *model,
	BSIM3v32instance **heres,
        VecNd *DrainSatCurrent,
        VecNd *SourceSatCurrent
)
{
	int	error;
	double dsat,ssat;
	for(int idx=0;idx<NSIMD;idx++)
	{
		error = BSIM3v32_ACM_saturationCurrents(
		      model, heres[idx],
		      &dsat,
		      &ssat
		);
		(*DrainSatCurrent)[idx] = dsat;
		(*SourceSatCurrent)[idx] = ssat;
		if(error) return error;
	}
	return error;
}

static inline int vecN_BSIM3v32_ACM_junctionCapacitances(
	BSIM3v32model *model,
	BSIM3v32instance **heres,
	VecNd *areaDrainBulkCapacitance,
	VecNd *periDrainBulkCapacitance,
	VecNd *gateDrainBulkCapacitance,
	VecNd *areaSourceBulkCapacitance,
	VecNd *periSourceBulkCapacitance,
	VecNd *gateSourceBulkCapacitance

)
{
	int	error;
	double areaDB,periDB,gateDB,areaSB,periSB,gateSB;
	
	for(int idx=0;idx<NSIMD;idx++)
	{
		error = BSIM3v32_ACM_junctionCapacitances(
		      model, heres[idx],
		      &areaDB,
		      &periDB,
		      &gateDB,
		      &areaSB,
		      &periSB,
		      &gateSB
		);
		(*areaDrainBulkCapacitance)[idx]=areaDB;
		(*periDrainBulkCapacitance)[idx]=periDB;
		(*gateDrainBulkCapacitance)[idx]=gateDB;
		(*areaSourceBulkCapacitance)[idx]=areaSB;
		(*periSourceBulkCapacitance)[idx]=periSB;
		(*gateSourceBulkCapacitance)[idx]=gateSB;
		if(error) return error;
	}
	return error;
}

#define vecNu_pow(x,p) vecN_exp(vecN_log(x)*p)

#if NSIMD==4
#define vec4_SIMDLOADDATA vecN_SIMDLOADDATA
#define vec4_BSIM3v32_ACM_saturationCurrents vecN_BSIM3v32_ACM_saturationCurrents
#define vec4_BSIM3v32_ACM_junctionCapacitances vecN_BSIM3v32_ACM_junctionCapacitances
#define vec4_NIintegrate vecN_NIintegrate
#define vec4_pow0p7(x,p) vecNu_pow(x,p)
#define vec4_powMJ(x,p) vecNu_pow(x,p)
#define vec4_powMJSW(x,p) vecNu_pow(x,p)
#define vec4_powMJSWG(x,p) vecNu_pow(x,p)
#define vec4_BSIM3v32_StateAccess vecN_StateAccess
#define vec4_BSIM3v32_StateStore vecN_StateStore
#define vec4_BSIM3v32_StateAdd vecN_StateAdd
#define vec4_BSIM3v32_StateSub vecN_StateSub
#endif

#if NSIMD==8
#define vec8_SIMDLOADDATA vecN_SIMDLOADDATA
#define vec8_BSIM3v32_ACM_saturationCurrents vecN_BSIM3v32_ACM_saturationCurrents
#define vec8_BSIM3v32_ACM_junctionCapacitances vecN_BSIM3v32_ACM_junctionCapacitances
#define vec8_NIintegrate vecNu_NIintegrate
#define vec8_pow0p7(x,p) vecNu_pow(x,p)
#define vec8_powMJ(x,p) vecNu_pow(x,p)
#define vec8_powMJSW(x,p) vecNu_pow(x,p)
#define vec8_powMJSWG(x,p) vecNu_pow(x,p)
#define vec8_BSIM3v32_StateAccess vecN_StateAccess
#define vec8_BSIM3v32_StateStore vecN_StateStore
#define vec8_BSIM3v32_StateAdd vecN_StateAdd
#define vec8_BSIM3v32_StateSub vecN_StateSub
#endif

#if NSIMD==2
#define vec2_SIMDLOADDATA vecN_SIMDLOADDATA
#define vec2_BSIM3v32_ACM_saturationCurrents vecN_BSIM3v32_ACM_saturationCurrents
#define vec2_BSIM3v32_ACM_junctionCapacitances vecN_BSIM3v32_ACM_junctionCapacitances
#define vec2_NIintegrate vecN_NIintegrate
#define vec2_pow0p7(x,p) vecNu_pow(x,p)
#define vec2_powMJ(x,p) vecNu_pow(x,p)
#define vec2_powMJSW(x,p) vecNu_pow(x,p)
#define vec2_powMJSWG(x,p) vecNu_pow(x,p)
#define vec2_BSIM3v32_StateAccess vecN_StateAccess
#define vec2_BSIM3v32_StateStore vecN_StateStore
#define vec2_BSIM3v32_StateAdd vecN_StateAdd
#define vec2_BSIM3v32_StateSub vecN_StateSub
#endif

int BSIM3v32LoadSIMD(BSIM3v32instance **heres, CKTcircuit *ckt
#ifndef USE_OMP
	, double data[7][NSIMD]
#endif
)
{
    BSIM3v32model *model = BSIM3v32modPtr(heres[0]);
    struct bsim3v32SizeDependParam *pParam;
    pParam = heres[0]->pParam; /* same of all NSIMD instances */

#if NSIMD==4
#ifdef USE_OMP
    #include "b3v32ldseq_simd4d_omp.c"
#else
    #include "b3v32ldseq_simd4d.c"
#endif
#elif NSIMD==8
#ifdef USE_OMP
    #include "b3v32ldseq_simd8d_omp.c"
#else
    #include "b3v32ldseq_simd8d.c"
#endif
#elif NSIMD==2
#ifdef USE_OMP
    #include "b3v32ldseq_simd2d_omp.c"
#else
    #include "b3v32ldseq_simd2d.c"
#endif
#else
#error Unsupported value for NSIMD
#endif
	
    return(OK);
	
}


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
 ****************************
/* draft version, not tested, not even compiled */

#include <math.h>
#include <x86intrin.h>
#include <signal.h>

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3v32def.h"
#include "b3v32acm.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/devdefs.h"
#include "ngspice/suffix.h"

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

#define NSIMD 8
#define USEX86INTRINSICS 1

typedef double Vec8d __attribute__ ((vector_size (sizeof(double)*NSIMD), aligned (sizeof(double)*NSIMD)));
typedef long int Vec8m __attribute__ ((vector_size (sizeof(double)*NSIMD), aligned (sizeof(double)*NSIMD)));


#define SIMDANY(err) (err!=0)
#define SIMDIFYCMD(cmd) /* empty */
#define SIMDifySaveScope(sc) /* empty */
#define SIMDVECTORMACRO(val) ((Vec8d) {val,val,val,val})
#define SIMDVECTOR(val) vec8_SIMDTOVECTOR(val)
#define SIMDVECTORMASK(val) vec8_SIMDTOVECTORMASK(val)


#ifdef USEX86INTRINSICS
#define vec8_MAX(a,b) _mm512_max_pd(a,b)
#define vec8_exp(a) _mm512_exp_pd(a)
#define vec8_log(a) _mm512_log_pd(a)
#define vec8_sqrt(a) _mm512_sqrt_pd(a)

static inline Vec8d vec8_blend(Vec8d fa, Vec8d tr, Vec8m mask)
{
	/* mask follow gcc vector extension comparison results false=0 true=-1 */
	/* so we can't use the new _mm512_mask_blendv_pd intrinsics */
	#define SAFER
	#ifdef SAFER
	/* support mask = false:0, true: 1 or -1 */
	return (Vec8d) _mm512_ternarylogic_epi64(
		_mm512_castpd_si512(fa),
		_mm512_castpd_si512(tr),
		_mm512_srai_epi64(_mm512_castpd_si512(mask), 63),
		0xd8);
	#else
	/* support only mask 0 or -1, mask=1 will fail, but should be OK with this code */
	return (Vec8d) _mm512_ternarylogic_epi64(
		_mm512_castpd_si512(fa),
		_mm512_castpd_si512(tr),
		_mm512_castpd_si512(mask),
		0xd8);
	#endif
	
	
}
static inline Vec8d vec8_fabs(Vec8d x)
{
	return (Vec8d) _mm512_abs_pd(x);
}
#else
#error X86 AVX512 instrinsics required for using SIMD8 version
#endif


/* some debug utils functions */
void vec8_printd(const char* msg, const char* name, Vec8d vecd)
{
	printf("%s %s %g %g %g %g\n",msg,name,vecd[0],vecd[1],vecd[2],vecd[3]);	
}

void vec8_printm(const char* msg, const char* name, Vec8m vecm)
{
	printf("%s %s %ld %ld %ld %ld\n",msg,name,vecm[0],vecm[1],vecm[2],vecm[3]);	
}

void vec8_CheckCollisions(Vec8m stateindexes, const char* msg)
{
	for(int i=0;i<NSIMD;i++)
	for(int j=0;j<NSIMD;j++)
	if(i!=j)
	if(stateindexes[i]==stateindexes[j])
	{
		printf("%s, collisions %ld %ld %ld %ld!\n",msg,stateindexes[0],stateindexes[1],stateindexes[2],stateindexes[3]);
		raise(SIGINT);
	}
}

/* useful vectorized functions */
static inline Vec8d SIMDLOADDATA(int idx, double data[7][NSIMD])
{
	return (Vec8d) {data[idx][0],data[idx][1],data[idx][2],data[idx][3],
		data[idx][4],data[idx][5],data[idx][6],data[idx][7]};
}

static inline Vec8d vec8_BSIM3v32_StateAccess(double* cktstate, Vec8m stateindexes)
{
	return (Vec8d) {
	 cktstate[stateindexes[0]],
	 cktstate[stateindexes[1]],
	 cktstate[stateindexes[2]],
	 cktstate[stateindexes[3]],
	 cktstate[stateindexes[4]],
	 cktstate[stateindexes[5]],
	 cktstate[stateindexes[6]],
	 cktstate[stateindexes[7]]
	};
}


static inline void vec8_BSIM3v32_StateStore(double* cktstate, Vec8m stateindexes, Vec8d values)
{
	if(0) vec8_CheckCollisions(stateindexes,"SateStore");
	for(int idx=0;idx<NSIMD;idx++)
	{
		cktstate[stateindexes[idx]] = values[idx];
	}
}

static inline void vec8_BSIM3v32_StateAdd(double* cktstate, Vec8m stateindexes, Vec8d values)
{
	if(0) vec8_CheckCollisions(stateindexes,"StateAdd");
	for(int idx=0;idx<NSIMD;idx++)
	{
		cktstate[stateindexes[idx]] += values[idx];
	}
}

static inline void vec8_BSIM3v32_StateSub(double* cktstate, Vec8m stateindexes, Vec8d values)
{
	if(0) vec8_CheckCollisions(stateindexes,"StateSub");
	for(int idx=0;idx<NSIMD;idx++)
	{
		cktstate[stateindexes[idx]] -= values[idx];
	}
}


static inline int vec8_BSIM3v32_ACM_saturationCurrents
(
	BSIM3v32model *model,
	BSIM3v32instance **heres,
        Vec8d *DrainSatCurrent,
        Vec8d *SourceSatCurrent
)
{
	int	error;
	for(int idx=0;idx<NSIMD;idx++)
	{
		error = BSIM3v32_ACM_saturationCurrents(
		      model, heres[idx],
		      &((*DrainSatCurrent)[idx]),
		      &((*SourceSatCurrent)[idx])
		);
		if(error) return error;
	}
	return error;
}

static inline int vec8_BSIM3v32_ACM_junctionCapacitances(
	BSIM3v32model *model,
	BSIM3v32instance **heres,
	Vec8d *areaDrainBulkCapacitance,
	Vec8d *periDrainBulkCapacitance,
	Vec8d *gateDrainBulkCapacitance,
	Vec8d *areaSourceBulkCapacitance,
	Vec8d *periSourceBulkCapacitance,
	Vec8d *gateSourceBulkCapacitance

)
{
	int	error;
	for(int idx=0;idx<NSIMD;idx++)
	{
		error = BSIM3v32_ACM_junctionCapacitances(
		      model, heres[idx],
		      &((*areaDrainBulkCapacitance)[idx]),
		      &((*periDrainBulkCapacitance)[idx]),
		      &((*gateDrainBulkCapacitance)[idx]),
		      &((*areaSourceBulkCapacitance)[idx]),
		      &((*periSourceBulkCapacitance)[idx]),
		      &((*gateSourceBulkCapacitance)[idx])
		);
		if(error) return error;
	}
	return error;
}

/* geq, ceq, and zero are not vectors because there are unused */
static inline int vec8_NIintegrate(CKTcircuit* ckt, double* geq, double *ceq, double zero, Vec8m chargestate)
{
	int	error;
	if (0) vec8_CheckCollisions(chargestate, "NIIntegrate");
	for(int idx=0;idx<NSIMD;idx++)
	{
		error = NIintegrate(ckt,geq,ceq,zero,chargestate[idx]);
		if(error) return error;
	}
	return error;
}

static inline int vec8_SIMDCOUNT(Vec8m mask) {
	return (mask[0] ? 1 : 0) + (mask[1] ? 1 : 0) + (mask[2] ? 1 : 0) + (mask[3] ? 1 : 0)
	 + (mask[4] ? 1 : 0) + (mask[5] ? 1 : 0) + (mask[6] ? 1 : 0) + (mask[7] ? 1 : 0);
}

static inline Vec8d vec8_SIMDTOVECTOR(double val)
{
	return (Vec8d) {val,val,val,val,val,val,val,val};
}
static inline Vec8m vec8_SIMDTOVECTORMASK(int val)
{
	return (Vec8m) {val,val,val,val,val,val,val,val};
}


int BSIM3v32LoadSIMD8(BSIM3v32instance **heres, CKTcircuit *ckt, double data[7][NSIMD]) {
    BSIM3v32model *model = BSIM3v32modPtr(heres[0]);
    if(0) printf("BSIM3v32LoadSIMD %s model %s\n", heres[0]->gen.GENname, model->gen.GENmodName);
    struct bsim3v32SizeDependParam *pParam;
    pParam = heres[0]->pParam; /* same of all NSIMD instances */

#if 1    
    #include "b3v32ldseq_simd8.c"
#endif
    
    return(OK);
	
}


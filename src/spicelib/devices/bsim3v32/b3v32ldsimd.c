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

#define USEX86INTRINSICS 1

typedef double Vec4d __attribute__ ((vector_size (sizeof(double)*NSIMD), aligned (sizeof(double)*NSIMD)));
typedef long int Vec4m __attribute__ ((vector_size (sizeof(double)*NSIMD), aligned (sizeof(double)*NSIMD)));


#define SIMDANY(err) (err!=0)
#define SIMDIFYCMD(cmd) /* empty */
#define SIMDifySaveScope(sc) /* empty */

#define vec4_pow0p7(x,p) vec4_mypow(x,p)
#define vec4_powMJ(x,p) vec4_mypow(x,p)
#define vec4_powMJSW(x,p) vec4_mypow(x,p)
#define vec4_powMJSWG(x,p) vec4_mypow(x,p)

#if USEX86INTRINSICS==1
/* libmvec prototypes */
/* Caution: those libmvec functions are not as precise as std libm */
__m256d _ZGVdN4v_exp(__m256d x);
__m256d _ZGVdN4v_log(__m256d x);

#define vec4_MAX(a,b) _mm256_max_pd(a,b)
#define vec4_exp(a) _ZGVdN4v_exp(a) 
#define vec4_log(a) _ZGVdN4v_log(a)
#define vec4_sqrt(a) _mm256_sqrt_pd(a)


static inline Vec4d vec4_blend(Vec4d fa, Vec4d tr, Vec4m mask)
{
	return _mm256_blendv_pd(fa,tr, (Vec4d) mask);
}

static inline Vec4d vec4_fabs(Vec4d x)
{
	return vec4_blend(x,-x,x<0);
}

#else
/* vector-libm prototypes */
Vec4d vec4_exp_vectorlibm(Vec4d x); /* defined in vec4_exp.c */
Vec4d vec4_log_vectorlibm(Vec4d x); /* defined in vec4_log.c */
#define vec4_exp(a) vec4_exp_vectorlibm(a)
#define vec4_log(a) vec4_log_vectorlibm(a)
static inline Vec4d vec4_MAX(Vec4d a, Vec4d b)
{
	return vec4_blend(a,b,a<b);
}
static inline Vec4d vec4_blend(Vec4d fa, Vec4d tr, Vec4m mask)
{
	/* hope for good vectorization by the compiler ! */
	Vec4d res;
	#pragma omp simd
	for(int i=0;i<4;i++)
	{
		res[i] = mask[i] ? tr[i] : fa[i];
	}
	return res;
}
static inline Vec4d vec4_fabs(Vec4d x)
{
	/* hope for good vectorization by the compiler ! */
	Vec4d res;
	#pragma omp simd
	for(int i=0;i<4;i++)
	{
		res[i] = (x[i] < 0) ? -x[i] : x[i];
	}
	return res;
}
static inline Vec4d vec4_sqrt(Vec4d x)
{
	/* hope for good vectorization by the compiler ! */
	Vec4d res;
	#pragma omp simd
	for(int i=0;i<4;i++)
	{
		res[i] = sqrt(x[i]);
	}
	return res;
}
#endif

static inline Vec4d vec4_mypow(Vec4d x, double p)
{
	return vec4_exp(vec4_log(x)*p);
}


/* some debug utils functions */
void vec4_printd(const char* msg, const char* name, Vec4d vecd)
{
	printf("%s %s %g %g %g %g\n",msg,name,vecd[0],vecd[1],vecd[2],vecd[3]);	
}

void vec4_printm(const char* msg, const char* name, Vec4m vecm)
{
	printf("%s %s %ld %ld %ld %ld\n",msg,name,vecm[0],vecm[1],vecm[2],vecm[3]);	
}

void vec4_CheckCollisions(Vec4m stateindexes, const char* msg)
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
static inline Vec4d SIMDLOADDATA(int idx, double data[7][NSIMD])
{
	return (Vec4d) {data[idx][0],data[idx][1],data[idx][2],data[idx][3]};
}

static inline Vec4d vec4_BSIM3v32_StateAccess(double* cktstate, Vec4m stateindexes)
{
	return (Vec4d) {
	 cktstate[stateindexes[0]],
	 cktstate[stateindexes[1]],
	 cktstate[stateindexes[2]],
	 cktstate[stateindexes[3]]
	};
}


static inline void vec4_BSIM3v32_StateStore(double* cktstate, Vec4m stateindexes, Vec4d values)
{
	if(0) vec4_CheckCollisions(stateindexes,"SateStore");
	for(int idx=0;idx<NSIMD;idx++)
	{
		cktstate[stateindexes[idx]] = values[idx];
	}
}

static inline void vec4_BSIM3v32_StateAdd(double* cktstate, Vec4m stateindexes, Vec4d values)
{
	if(0) vec4_CheckCollisions(stateindexes,"StateAdd");
	for(int idx=0;idx<NSIMD;idx++)
	{
		cktstate[stateindexes[idx]] += values[idx];
	}
}

static inline void vec4_BSIM3v32_StateSub(double* cktstate, Vec4m stateindexes, Vec4d values)
{
	if(0) vec4_CheckCollisions(stateindexes,"StateSub");
	for(int idx=0;idx<NSIMD;idx++)
	{
		cktstate[stateindexes[idx]] -= values[idx];
	}
}

static inline Vec4d vec4_exp_seq(Vec4d val)
{
	return (Vec4d) {exp(val[0]),exp(val[1]),exp(val[2]),exp(val[3])};
}
static inline Vec4d vec4_log_seq(Vec4d val)
{
	return (Vec4d) {log(val[0]),log(val[1]),log(val[2]),log(val[3])};
}
static inline Vec4d vec4_sqrt_seq(Vec4d val)
{
	return (Vec4d) {sqrt(val[0]),sqrt(val[1]),sqrt(val[2]),sqrt(val[3])};
}
static inline Vec4d vec4_MAX_seq(Vec4d a, Vec4d b)
{
	return (Vec4d) {MAX(a[0],b[0]),MAX(a[1],b[1]),MAX(a[2],b[2]),MAX(a[3],b[3])};
}

static inline int vec4_BSIM3v32_ACM_saturationCurrents
(
	BSIM3v32model *model,
	BSIM3v32instance **heres,
        Vec4d *DrainSatCurrent,
        Vec4d *SourceSatCurrent
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

static inline int vec4_BSIM3v32_ACM_junctionCapacitances(
	BSIM3v32model *model,
	BSIM3v32instance **heres,
	Vec4d *areaDrainBulkCapacitance,
	Vec4d *periDrainBulkCapacitance,
	Vec4d *gateDrainBulkCapacitance,
	Vec4d *areaSourceBulkCapacitance,
	Vec4d *periSourceBulkCapacitance,
	Vec4d *gateSourceBulkCapacitance

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

/* geq, ceq, and zero are not translated to vectors because there are unused */
static inline int vec4_NIintegrate(CKTcircuit* ckt, double* geq, double *ceq, double zero, Vec4m chargestate)
{
	int	error;
	if (0) vec4_CheckCollisions(chargestate, "NIIntegrate");
	for(int idx=0;idx<NSIMD;idx++)
	{
		error = NIintegrate(ckt,geq,ceq,zero,chargestate[idx]);
		if(error) return error;
	}
	return error;
}

static inline int vec4_SIMDCOUNT(Vec4m mask) {
	return (mask[0] ? 1 : 0) + (mask[1] ? 1 : 0) + (mask[2] ? 1 : 0) + (mask[3] ? 1 : 0);
}

static inline Vec4d vec4_SIMDTOVECTOR(double val)
{
	return (Vec4d) {val,val,val,val};
}
static inline Vec4m vec4_SIMDTOVECTORMASK(int val)
{
	return (Vec4m) {val,val,val,val};
}


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
    #pragma message "Use OMP SIMD4 version"
    #include "b3v32ldseq_simd4_omp.c"
#else
    #include "b3v32ldseq_simd4.c"
#endif
#elif NSIMD==8
#ifdef USE_OMP
    #pragma message "Use OMP SIMD8 version"
    #include "b3v32ldseq_simd8_omp.c"
#else
    #include "b3v32ldseq_simd8.c"
#endif
#else
#error Unsupported value for NSIMD
#endif
	
    return(OK);
	
}


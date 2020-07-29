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

/* disable omp simd for GCC, as it slow down a bit */
#if !defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)
#define USE_OMPSIMD
#endif

/* HAS_LIBMVEC defined from configure.ac */

/* USE_SERIAL_FORM can be defined but has no performance influence */

/******* vec4_blend *******/
#if USEX86INTRINSICS==1
static inline Vec4d vec4_blend(Vec4d fa, Vec4d tr, Vec4m mask)
{
	return _mm256_blendv_pd(fa,tr, (Vec4d) mask);
}
#else
static inline Vec4d vec4_blend(Vec4d fa, Vec4d tr, Vec4m mask)
{
	Vec4d r;
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(4)
	#endif
	for(int i=0;i<4;i++)
		r[i] = (mask[i]==0 ? fa[i] : tr[i]);
	return r;
}
#endif

/******* vec4_exp, vec4_log *******/
#ifdef HAS_LIBMVEC
Vec4d _ZGVdN4v_exp(Vec4d x);
Vec4d _ZGVdN4v_log(Vec4d x);
/*Vec4d _ZGVdN4vv_pow(Vec4d x, Vec4d y);*/

#define vec4_exp(a) _ZGVdN4v_exp(a) 
#define vec4_log(a) _ZGVdN4v_log(a)
#endif

#ifndef HAS_LIBMVEC
static inline Vec4d vec4_exp(Vec4d x)
{
	Vec4d r;
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(4)
	#endif
	for(int i=0;i<4;i++)
		r[i] = exp(x[i]);
	return r;
}

static inline Vec4d vec4_log(Vec4d x)
{
	Vec4d r;
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(4)
	#endif
	for(int i=0;i<4;i++)
		r[i] = log(x[i]);
	return r;
}
#endif

/******* vec4_MAX, vec4_sqrt *******/
#ifdef USEX86INTRINSICS
#define vec4_MAX(a,b) _mm256_max_pd(a,b)
#define vec4_sqrt(a) _mm256_sqrt_pd(a)
#else
static inline Vec4d vec4_MAX(Vec4d x, Vec4d y)
{
	Vec4d r;
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(4)
	#endif
	for(int i=0;i<4;i++)
		r[i] = MAX(x[i],y[i]);
	return r;
}

static inline Vec4d vec4_sqrt(Vec4d x)
{
	Vec4d r;
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(4)
	#endif
	for(int i=0;i<4;i++)
		r[i] = sqrt(x[i]);
	return r;
}
#endif

/******* vec4_fabs *******/
#ifdef USE_SERIAL_FORM
static inline Vec4d vec4_fabs(Vec4d x)
{
	return vec4_blend(x,-x,x<0);
}
#else
static inline Vec4d vec4_fabs(Vec4d x)
{
	Vec4d r;
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(4)
	#endif
	for(int i=0;i<4;i++)
		r[i] = fabs(x[i]);
	return r;
}
#endif

#define vec4_pow0p7(x,p) vec4_pow(x,p)
#define vec4_powMJ(x,p) vec4_pow(x,p)
#define vec4_powMJSW(x,p) vec4_pow(x,p)
#define vec4_powMJSWG(x,p) vec4_pow(x,p)

static inline Vec4d vec4_pow(Vec4d x, double p)
{
	/*return _ZGVdN4vv_pow(x,(Vec4d) {p,p,p,p});*/
	return vec4_exp(vec4_log(x)*p);
}

/******* vec4_SIMDTOVECTOR, vec4_SIMDTOVECTORMASK *******/
#ifdef USE_SERIAL_FORM
static inline Vec4d vec4_SIMDTOVECTOR(double val)
{
	return (Vec4d) {val,val,val,val};
}

static inline Vec4m vec4_SIMDTOVECTORMASK(int val)
{
	return (Vec4m) {val,val,val,val};
}

#else
static inline Vec4d vec4_SIMDTOVECTOR(double val)
{
	Vec4d r;
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(4)
	#endif
	for(int i=0;i<4;i++)
		r[i] = val;
	return r;
}

static inline Vec4m vec4_SIMDTOVECTORMASK(int val)
{
	Vec4m r;
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(4)
	#endif
	for(int i=0;i<4;i++)
		r[i] = val;
	return r;
}
#endif


static inline Vec4d vec4_SIMDLOADDATA(int idx, double data[7][4])
{
	return (Vec4d) {data[idx][0],data[idx][1],data[idx][2],data[idx][3]};
}

/******* vec4_BSIM3v32_StateAccess *******/
#ifdef USE_SERIAL_FORM
static inline Vec4d vec4_BSIM3v32_StateAccess(double* cktstate, Vec4m stateindexes)
{
	return (Vec4d) {
	 cktstate[stateindexes[0]],
	 cktstate[stateindexes[1]],
	 cktstate[stateindexes[2]],
	 cktstate[stateindexes[3]]
	};
}
#else
static inline Vec4d vec4_BSIM3v32_StateAccess(double* cktstate, Vec4m stateindexes)
{
	Vec4d r;
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(4)
	#endif
	for(int i=0;i<4;i++)
		r[i] =  cktstate[stateindexes[i]];
	return r;
}
#endif

static inline void vec4_BSIM3v32_StateStore(double* cktstate, Vec4m stateindexes, Vec4d values)
{
	/*if(0) vec4_CheckCollisions(stateindexes,"SateStore");*/
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(4)
	#endif
	for(int idx=0;idx<4;idx++)
	{
		cktstate[stateindexes[idx]] = values[idx];
	}
}

static inline void vec4_BSIM3v32_StateAdd(double* cktstate, Vec4m stateindexes, Vec4d values)
{
	/*if(0) vec4_CheckCollisions(stateindexes,"StateAdd");*/
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(4)
	#endif
	for(int idx=0;idx<4;idx++)
	{
		cktstate[stateindexes[idx]] += values[idx];
	}
}

static inline void vec4_BSIM3v32_StateSub(double* cktstate, Vec4m stateindexes, Vec4d values)
{
	/*if(0) vec4_CheckCollisions(stateindexes,"StateSub");*/
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(4)
	#endif
	for(int idx=0;idx<4;idx++)
	{
		cktstate[stateindexes[idx]] -= values[idx];
	}
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
	for(int idx=0;idx<4;idx++)
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
	
	for(int idx=0;idx<4;idx++)
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
	/*if (0) vec4_CheckCollisions(chargestate, "NIIntegrate");*/
	for(int idx=0;idx<4;idx++)
	{
		error = NIintegrate(ckt,geq,ceq,zero,chargestate[idx]);
		if(error) return error;
	}
	return error;
}

static inline int vec4_SIMDCOUNT(Vec4m mask) {
	return (mask[0] ? 1 : 0) + (mask[1] ? 1 : 0) + (mask[2] ? 1 : 0) + (mask[3] ? 1 : 0);
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
	for(int i=0;i<4;i++)
	for(int j=0;j<4;j++)
	if(i!=j)
	if(stateindexes[i]==stateindexes[j])
	{
		printf("%s, collisions %ld %ld %ld %ld!\n",msg,stateindexes[0],stateindexes[1],stateindexes[2],stateindexes[3]);
		raise(SIGINT);
	}
}

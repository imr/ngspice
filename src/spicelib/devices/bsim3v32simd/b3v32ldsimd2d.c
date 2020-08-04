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


static inline Vec2d vec2_blend(Vec2d fa, Vec2d tr, Vec2m mask)
{
	Vec2d r;
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(4)
	#endif
	for(int i=0;i<2;i++)
		r[i] = (mask[i]==0 ? fa[i] : tr[i]);
	return r;
}

static inline Vec2d vec2_exp(Vec2d x)
{
	Vec2d r;
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(4)
	#endif
	for(int i=0;i<2;i++)
		r[i] = exp(x[i]);
	return r;
}

static inline Vec2d vec2_log(Vec2d x)
{
	Vec2d r;
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(4)
	#endif
	for(int i=0;i<2;i++)
		r[i] = log(x[i]);
	return r;
}

static inline Vec2d vec2_max(Vec2d x, Vec2d y)
{
	Vec2d r;
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(4)
	#endif
	for(int i=0;i<2;i++)
		r[i] = MAX(x[i],y[i]);
	return r;
}

static inline Vec2d vec2_sqrt(Vec2d x)
{
	Vec2d r;
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(4)
	#endif
	for(int i=0;i<2;i++)
		r[i] = sqrt(x[i]);
	return r;
}

static inline Vec2d vec2_fabs(Vec2d x)
{
	Vec2d r;
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(4)
	#endif
	for(int i=0;i<2;i++)
		r[i] = fabs(x[i]);
	return r;
}

#define vec2_pow0p7(x,p) vec2_pow(x,p)
#define vec2_powMJ(x,p) vec2_pow(x,p)
#define vec2_powMJSW(x,p) vec2_pow(x,p)
#define vec2_powMJSWG(x,p) vec2_pow(x,p)

static inline Vec2d vec2_pow(Vec2d x, double p)
{
	return vec2_exp(vec2_log(x)*p);
}

/* useful vectorized functions */
static inline Vec2d vec2_SIMDTOVECTOR(double val)
{
	return (Vec2d) {val,val};
}

static inline Vec2m vec2_SIMDTOVECTORMASK(int32_t val)
{
	return (Vec2m) {val,val};
}

static inline Vec2d vec2_SIMDLOADDATA(int idx, double data[7][2])
{
	return (Vec2d) {data[idx][0],data[idx][1]};
}

static inline Vec2d vec2_BSIM3v32_StateAccess(double* cktstate, Vec2m stateindexes)
{
	Vec2d r;
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(4)
	#endif
	for(int i=0;i<2;i++)
		r[i] =  cktstate[stateindexes[i]];
	return r;
}


static inline void vec2_BSIM3v32_StateStore(double* cktstate, Vec2m stateindexes, Vec2d values)
{
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(4)
	#endif
	for(int idx=0;idx<2;idx++)
	{
		cktstate[stateindexes[idx]] = values[idx];
	}
}

static inline void vec2_BSIM3v32_StateAdd(double* cktstate, Vec2m stateindexes, Vec2d values)
{
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(4)
	#endif
	for(int idx=0;idx<2;idx++)
	{
		cktstate[stateindexes[idx]] += values[idx];
	}
}

static inline void vec2_BSIM3v32_StateSub(double* cktstate, Vec2m stateindexes, Vec2d values)
{
	#ifdef USE_OMPSIMD
	#pragma omp simd simdlen(4)
	#endif
	for(int idx=0;idx<2;idx++)
	{
		cktstate[stateindexes[idx]] -= values[idx];
	}
}

static inline int vec2_BSIM3v32_ACM_saturationCurrents
(
	BSIM3v32model *model,
	BSIM3v32instance **heres,
        Vec2d *DrainSatCurrent,
        Vec2d *SourceSatCurrent
)
{
	int	error;
	double dsat,ssat;
	for(int idx=0;idx<2;idx++)
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

static inline int vec2_BSIM3v32_ACM_junctionCapacitances(
	BSIM3v32model *model,
	BSIM3v32instance **heres,
	Vec2d *areaDrainBulkCapacitance,
	Vec2d *periDrainBulkCapacitance,
	Vec2d *gateDrainBulkCapacitance,
	Vec2d *areaSourceBulkCapacitance,
	Vec2d *periSourceBulkCapacitance,
	Vec2d *gateSourceBulkCapacitance

)
{
	int	error;
	double areaDB,periDB,gateDB,areaSB,periSB,gateSB;
	
	for(int idx=0;idx<2;idx++)
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
static inline int vec2_NIintegrate(CKTcircuit* ckt, double* geq, double *ceq, double zero, Vec2m chargestate)
{
	int	error;
	for(int idx=0;idx<2;idx++)
	{
		error = NIintegrate(ckt,geq,ceq,zero,chargestate[idx]);
		if(error) return error;
	}
	return error;
}

static inline int vec2_SIMDCOUNT(Vec2m mask) {
	return (mask[0] ? 1 : 0) + (mask[1] ? 1 : 0);
}

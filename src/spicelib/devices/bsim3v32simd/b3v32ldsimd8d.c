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

static inline Vec8d vec8_blend(Vec8d fa, Vec8d tr, Vec8m mask)
{
	Vec8d r;
	#pragma omp simd
	for(int i=0;i<8;i++)
		r[i] = (mask[i]==0 ? fa[i] : tr[i]);
	return r;
}

static inline Vec8d vec8_exp(Vec8d x)
{
	Vec8d r;
	#pragma omp simd
	for(int i=0;i<8;i++)
		r[i] = exp(x[i]);
	return r;
}

static inline Vec8d vec8_log(Vec8d x)
{
	Vec8d r;
	#pragma omp simd
	for(int i=0;i<8;i++)
		r[i] = log(x[i]);
	return r;
}

static inline Vec8d vec8_max(Vec8d x, Vec8d y)
{
	Vec8d r;
	#pragma omp simd
	for(int i=0;i<8;i++)
		r[i] = MAX(x[i],y[i]);
	return r;
}

static inline Vec8d vec8_sqrt(Vec8d x)
{
	Vec8d r;
	#pragma omp simd
	for(int i=0;i<8;i++)
		r[i] = sqrt(x[i]);
	return r;
}

static inline Vec8d vec8_fabs(Vec8d x)
{
	Vec8d r;
	#pragma omp simd
	for(int i=0;i<8;i++)
		r[i] = fabs(x[i]);
	return r;
}

#define vec8_pow0p7(x,p) vec8_pow(x,p)
#define vec8_powMJ(x,p) vec8_pow(x,p)
#define vec8_powMJSW(x,p) vec8_pow(x,p)
#define vec8_powMJSWG(x,p) vec8_pow(x,p)

static inline Vec8d vec8_pow(Vec8d x, double p)
{
	return vec8_exp(vec8_log(x)*p);
}

/* useful vectorized functions */
static inline Vec8d vec8_SIMDTOVECTOR(double val)
{
	return (Vec8d) {val,val,val,val,val,val,val,val};
}

static inline Vec8m vec8_SIMDTOVECTORMASK(int32_t val)
{
	return (Vec8m) {val,val,val,val,val,val,val,val};
}

static inline Vec8d vec8_SIMDLOADDATA(int idx, double data[7][8])
{
	return (Vec8d) {data[idx][0],data[idx][1],data[idx][2],data[idx][3],data[idx][4],data[idx][5],data[idx][6],data[idx][7]};
}

static inline Vec8d vec8_BSIM3v32_StateAccess(double* cktstate, Vec8m stateindexes)
{
	Vec8d r;
	#pragma omp simd
	for(int i=0;i<8;i++)
		r[i] =  cktstate[stateindexes[i]];
	return r;
}


static inline void vec8_BSIM3v32_StateStore(double* cktstate, Vec8m stateindexes, Vec8d values)
{
	#pragma omp simd
	for(int idx=0;idx<8;idx++)
	{
		cktstate[stateindexes[idx]] = values[idx];
	}
}

static inline void vec8_BSIM3v32_StateAdd(double* cktstate, Vec8m stateindexes, Vec8d values)
{
	#pragma omp simd
	for(int idx=0;idx<8;idx++)
	{
		cktstate[stateindexes[idx]] += values[idx];
	}
}

static inline void vec8_BSIM3v32_StateSub(double* cktstate, Vec8m stateindexes, Vec8d values)
{
	#pragma omp simd
	for(int idx=0;idx<8;idx++)
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
	double dsat,ssat;
	for(int idx=0;idx<8;idx++)
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
	double areaDB,periDB,gateDB,areaSB,periSB,gateSB;
	
	for(int idx=0;idx<8;idx++)
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
static inline int vec8_NIintegrate(CKTcircuit* ckt, double* geq, double *ceq, double zero, Vec8m chargestate)
{
	int	error;
	for(int idx=0;idx<8;idx++)
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


#if 0
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
	for(int i=0;i<8;i++)
	for(int j=0;j<8;j++)
	if(i!=j)
	if(stateindexes[i]==stateindexes[j])
	{
		printf("%s, collisions %ld %ld %ld %ld!\n",msg,stateindexes[0],stateindexes[1],stateindexes[2],stateindexes[3]);
		raise(SIGINT);
	}
}
#endif

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

#if USEX86INTRINSICS==1
#include <x86intrin.h>
#endif

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

#if NSIMD==4
#include "b3v32ldsimd4d.c"
#endif

#if NSIMD==8
#include "b3v32ldsimd8d.c"
#endif

#if NSIMD==2
#include "b3v32ldsimd2d.c"
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


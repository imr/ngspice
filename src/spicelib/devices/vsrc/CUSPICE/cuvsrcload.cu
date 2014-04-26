/*
 * Copyright (c) 2014, NVIDIA Corporation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation and/or
 *    other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors may be used to
 *    endorse or promote products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "ngspice/config.h"
#include "ngspice/CUSPICE/cuniinteg.cuh"
#include "vsrcdefs.h"



/*** STUFF NEEDED BECAUSE OF SOME INCLUSIONS IN NGSPICE THAT ARE NOT AVAILABLE IN CUDA ***/
/* TRNOISE and TRRANDOM don't work in the CUDA implementation */

/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
**********/

#include <assert.h>
#include <stdint.h>

#ifdef _MSC_VER
#define llabs(x) ((x) < 0 ? -(x) : (x))
#endif

#define int64_min (((int64_t) -1) << 63)

#define TRUE 1
#define FALSE 0

/* From Bruce Dawson, Comparing floating point numbers,
   http://www.cygnus-software.com/papers/comparingfloats/Comparing%20floating%20point%20numbers.htm
   Original this function is named AlmostEqual2sComplement but we leave it to AlmostEqualUlps
   and can leave the code (measure.c, dctran.c) unchanged. The transformation to the 2's complement
   prevent problems around 0.0.
   One Ulp is equivalent to a maxRelativeError of between 1/4,000,000,000,000,000 and 1/8,000,000,000,000,000.
   Practical: 3 < maxUlps < some hundred's (or thousand's) - depending on numerical requirements.
*/

__device__
static
bool
AlmostEqualUlps (double A, double B, int maxUlps)
{
    int64_t aInt, bInt, intDiff;

    if (A == B)
        return TRUE ;

    /* If not - the entire method can not work */
    assert (sizeof(double) == sizeof(int64_t)) ;

    /* Make sure maxUlps is non-negative and small enough that the */
    /* default NAN won't compare as equal to anything. */
    assert (maxUlps > 0 && maxUlps < 4 * 1024 * 1024) ;

    aInt = *(int64_t*)&A ;
    /* Make aInt lexicographically ordered as a twos-complement int */
    if (aInt < 0)
        aInt = int64_min - aInt ;

    bInt = *(int64_t*)&B ;
    /* Make bInt lexicographically ordered as a twos-complement int */
    if (bInt < 0)
        bInt = int64_min - bInt ;

    intDiff = llabs (aInt - bInt) ;

/* printf("A:%e B:%e aInt:%d bInt:%d  diff:%d\n", A, B, aInt, bInt, intDiff); */

    if (intDiff <= maxUlps)
        return TRUE ;
    return FALSE ;
}



/*** CODE STARTING ***/
extern "C"
__global__ void cuVSRCload_kernel (VSRCparamGPUstruct, int, double, double, double, double, int, int *, double *, int *, double *) ;

extern "C"
int
cuVSRCload
(
GENmodel *inModel, CKTcircuit *ckt
)
{
    VSRCmodel *model = (VSRCmodel *)inModel ;
    int thread_x, thread_y, block_x ;

    cudaError_t status ;

    /*  loop through all the inductor models */
    for ( ; model != NULL ; model = VSRCnextModel(model))
    {
        /* Determining how many blocks should exist in the kernel */
        thread_x = 1 ;
        thread_y = 256 ;
        if (model->n_instances % thread_y != 0)
            block_x = (int)((model->n_instances + thread_y - 1) / thread_y) ;
        else
            block_x = model->n_instances / thread_y ;

        dim3 thread (thread_x, thread_y) ;

        /* Kernel launch */
        status = cudaGetLastError () ; // clear error status

        cuVSRCload_kernel <<< block_x, thread >>> (model->VSRCparamGPU, ckt->CKTmode, ckt->CKTtime,
                                                   ckt->CKTstep, ckt->CKTfinalTime, ckt->CKTsrcFact,
                                                   model->n_instances, model->d_PositionVector,
                                                   ckt->d_CKTloadOutput, model->d_PositionVectorRHS,
                                                   ckt->d_CKTloadOutputRHS) ;

        cudaDeviceSynchronize () ;

        status = cudaGetLastError () ; // check for launch error
        if (status != cudaSuccess)
        {
            fprintf (stderr, "Kernel launch failure in the Voltage Source Model\n\n") ;
            return (E_NOMEM) ;
        }
    }

    return (OK) ;
}

extern "C"
__global__
void
cuVSRCload_kernel
(
VSRCparamGPUstruct VSRCentry, int CKTmode, double CKTtime,
double CKTstep, double CKTfinalTime, double CKTsrcFact, int n_instances,
int *d_PositionVector, double *d_CKTloadOutput, int *d_PositionVectorRHS, double *d_CKTloadOutputRHS
)
{
    int instance_ID ;
    double time, value = 0.0 ;

    instance_ID = threadIdx.y + blockDim.y * blockIdx.x ;

    if (instance_ID < n_instances)
    {
        if (threadIdx.x == 0)
        {
            d_CKTloadOutput [d_PositionVector [instance_ID]] = 1.0 ;

            if ((CKTmode & (MODEDCOP | MODEDCTRANCURVE)) && VSRCentry.d_VSRCdcGivenArray [instance_ID])
            {
                /* load using DC value */
#ifdef XSPICE_EXP
/* gtri - begin - wbk - modify to process srcFact, etc. for all sources */
                value = VSRCentry.d_VSRCdcvalueArray [instance_ID] ;
#else
                value = VSRCentry.d_VSRCdcvalueArray [instance_ID] * CKTsrcFact ;
#endif
            } else {
                if (CKTmode & (MODEDC))
                    time = 0 ;
                else
                    time = CKTtime ;

                /* use the transient functions */
                switch (VSRCentry.d_VSRCfunctionTypeArray [instance_ID])
                {
                    default:
                        value = VSRCentry.d_VSRCdcvalueArray [instance_ID] ;
                        break ;

                    case PULSE:
                    {
                        double V1, V2, TD, TR, TF, PW, PER, basetime = 0.0 ;
#ifdef XSPICE
                        double PHASE, phase, deltat ;
#endif
                        V1 = VSRCentry.d_VSRCcoeffsArray [instance_ID] [0] ;
                        V2 = VSRCentry.d_VSRCcoeffsArray [instance_ID] [1] ;
                        TD = VSRCentry.d_VSRCfunctionOrderArray [instance_ID] > 2
                                                ? VSRCentry.d_VSRCcoeffsArray [instance_ID] [2] : 0.0 ;
                        TR = VSRCentry.d_VSRCfunctionOrderArray [instance_ID] > 3
                                                && VSRCentry.d_VSRCcoeffsArray [instance_ID] [3] != 0.0
                                                ? VSRCentry.d_VSRCcoeffsArray [instance_ID] [3] : CKTstep ;
                        TF = VSRCentry.d_VSRCfunctionOrderArray [instance_ID] > 4
                                                && VSRCentry.d_VSRCcoeffsArray [instance_ID] [4] != 0.0
                                                ? VSRCentry.d_VSRCcoeffsArray [instance_ID] [4] : CKTstep ;
                        PW = VSRCentry.d_VSRCfunctionOrderArray [instance_ID] > 5
                                                && VSRCentry.d_VSRCcoeffsArray [instance_ID] [5] != 0.0
                                                ? VSRCentry.d_VSRCcoeffsArray [instance_ID] [5] : CKTfinalTime ;
                        PER = VSRCentry.d_VSRCfunctionOrderArray [instance_ID] > 6
                                                && VSRCentry.d_VSRCcoeffsArray [instance_ID] [6] != 0.0
                                                ? VSRCentry.d_VSRCcoeffsArray [instance_ID] [6] : CKTfinalTime ;

                        /* shift time by delay time TD */
                        time -=  TD ;

#ifdef XSPICE
/* gtri - begin - wbk - add PHASE parameter */

                        PHASE = VSRCentry.d_VSRCfunctionOrderArray [instance_ID] > 7
                                                ? VSRCentry.d_VSRCcoeffsArray [instance_ID] [7] : 0.0 ;

                        /* normalize phase to cycles */
                        phase = PHASE / 360.0 ;
                        phase = fmod (phase, 1.0) ;
                        deltat = phase * PER ;
                        while (deltat > 0)
                            deltat -= PER ;

                        /* shift time by pase (neg. for pos. phase value) */
                        time += deltat ;

/* gtri - end - wbk - add PHASE parameter */
#endif
                        if (time > PER)
                        {
                            /* repeating signal - figure out where we are */
                            /* in period */
                            basetime = PER * floor (time / PER) ;
                            time -= basetime ;
                        }

                        if (time <= 0 || time >= TR + PW + TF)
                            value = V1 ;
                        else  if (time >= TR && time <= TR + PW)
                            value = V2 ;
                        else if (time > 0 && time < TR)
                            value = V1 + (V2 - V1) * time / TR ;
                        else /* time > TR + PW && < TR + PW + TF */
                            value = V2 + (V1 - V2) * (time - (TR + PW)) / TF ;
                    }
                    break ;

                    case SINE:
                    {
                        double VO, VA, FREQ, TD, THETA ;

#ifdef XSPICE
/* gtri - begin - wbk - add PHASE parameter */

                        double PHASE, phase ;

                        PHASE = VSRCentry.d_VSRCfunctionOrderArray [instance_ID] > 5
                                                ? VSRCentry.d_VSRCcoeffsArray [instance_ID] [5] : 0.0 ;

                        /* compute phase in radians */
                        phase = PHASE * M_PI / 180.0 ;
#endif

                        VO = VSRCentry.d_VSRCcoeffsArray [instance_ID] [0] ;
                        VA = VSRCentry.d_VSRCcoeffsArray [instance_ID] [1] ;
                        FREQ = VSRCentry.d_VSRCfunctionOrderArray [instance_ID] > 2
                                                 && VSRCentry.d_VSRCcoeffsArray [instance_ID] [2] != 0.0
                                                 ? VSRCentry.d_VSRCcoeffsArray [instance_ID] [2] : (1 / CKTfinalTime) ;
                        TD = VSRCentry.d_VSRCfunctionOrderArray [instance_ID] > 3
                                                 ? VSRCentry.d_VSRCcoeffsArray [instance_ID] [3] : 0.0 ;
                        THETA = VSRCentry.d_VSRCfunctionOrderArray [instance_ID] > 4
                                                 ? VSRCentry.d_VSRCcoeffsArray [instance_ID] [4] : 0.0 ;

                        time -= TD ;
                        if (time <= 0)
                        {

#ifdef XSPICE
                            value = VO + VA * sin (phase) ;
                        } else {
                            value = VO + VA * sin (FREQ * time * 2.0 * M_PI + phase) * exp (-time * THETA) ;
#else
                            value = VO ;
                        } else {
                            value = VO + VA * sin (FREQ * time * 2.0 * M_PI) * exp (-time * THETA) ;
/* gtri - end - wbk - add PHASE parameter */
#endif

                        }
                    }
                    break ;

                    case EXP:
                    {
                        double V1, V2, TD1, TD2, TAU1, TAU2 ;

                        V1 = VSRCentry.d_VSRCcoeffsArray [instance_ID] [0] ;
                        V2 = VSRCentry.d_VSRCcoeffsArray [instance_ID] [1] ;
                        TD1 = VSRCentry.d_VSRCfunctionOrderArray [instance_ID] > 2
                                                && VSRCentry.d_VSRCcoeffsArray [instance_ID] [2] != 0.0
                                                ? VSRCentry.d_VSRCcoeffsArray [instance_ID] [2] : CKTstep ;
                        TAU1 = VSRCentry.d_VSRCfunctionOrderArray [instance_ID] > 3
                                                && VSRCentry.d_VSRCcoeffsArray [instance_ID] [3] != 0.0
                                                ? VSRCentry.d_VSRCcoeffsArray [instance_ID] [3] : CKTstep ;
                        TD2 = VSRCentry.d_VSRCfunctionOrderArray [instance_ID] > 4
                                                && VSRCentry.d_VSRCcoeffsArray [instance_ID] [4] != 0.0
                                                ? VSRCentry.d_VSRCcoeffsArray [instance_ID] [4] : TD1 + CKTstep ;
                        TAU2 = VSRCentry.d_VSRCfunctionOrderArray [instance_ID] > 5
                                                && VSRCentry.d_VSRCcoeffsArray [instance_ID] [5]
                                                ? VSRCentry.d_VSRCcoeffsArray [instance_ID] [5] : CKTstep ;

                        if(time <= TD1)
                            value = V1 ;
                        else if (time <= TD2)
                            value = V1 + (V2 - V1) * (1 - exp (-(time - TD1) / TAU1)) ;
                        else
                            value = V1 + (V2 - V1) * (1 - exp (-(time - TD1) / TAU1))
                                       + (V1 - V2) * (1 - exp (-(time - TD2) / TAU2)) ;
                    }
                    break ;

                    case SFFM:
                    {
                        double VO, VA, FC, MDI, FS ;

#ifdef XSPICE
/* gtri - begin - wbk - add PHASE parameters */
                        double PHASEC, PHASES, phasec, phases ;

                        PHASEC = VSRCentry.d_VSRCfunctionOrderArray [instance_ID] > 5
                                                ? VSRCentry.d_VSRCcoeffsArray [instance_ID] [5] : 0.0 ;
                        PHASES = VSRCentry.d_VSRCfunctionOrderArray [instance_ID] > 6
                                                ? VSRCentry.d_VSRCcoeffsArray [instance_ID] [6] : 0.0 ;

                        /* compute phases in radians */
                        phasec = PHASEC * M_PI / 180.0 ;
                        phases = PHASES * M_PI / 180.0 ;
#endif

                        VO = VSRCentry.d_VSRCcoeffsArray [instance_ID] [0] ;
                        VA = VSRCentry.d_VSRCcoeffsArray [instance_ID] [1] ;
                        FC = VSRCentry.d_VSRCfunctionOrderArray [instance_ID] > 2
                                               && VSRCentry.d_VSRCcoeffsArray [instance_ID] [2]
                                               ? VSRCentry.d_VSRCcoeffsArray [instance_ID] [2] : (1 / CKTfinalTime) ;
                        MDI = VSRCentry.d_VSRCfunctionOrderArray [instance_ID] > 3
                                               ? VSRCentry.d_VSRCcoeffsArray [instance_ID] [3] : 0.0 ;
                        FS = VSRCentry.d_VSRCfunctionOrderArray [instance_ID] > 4
                                               && VSRCentry.d_VSRCcoeffsArray [instance_ID] [4]
                                               ? VSRCentry.d_VSRCcoeffsArray [instance_ID] [4] : (1 / CKTfinalTime) ;

#ifdef XSPICE
                        /* compute waveform value */
                        value = VO + VA * sin ((2.0 * M_PI * FC * time + phasec) +
                                MDI * sin (2.0 * M_PI * FS * time + phases)) ;
#else
                        value = VO + VA * sin ((2.0 * M_PI * FC * time) +
                                MDI * sin (2.0 * M_PI * FS * time)) ;
/* gtri - end - wbk - add PHASE parameters */
#endif

                    }
                    break ;

                    case AM:
                    {
                        double VA, FC, MF, VO, TD ;

#ifdef XSPICE
/* gtri - begin - wbk - add PHASE parameters */
                        double PHASEC, PHASES, phasec, phases ;

                        PHASEC = VSRCentry.d_VSRCfunctionOrderArray [instance_ID] > 5
                                                ? VSRCentry.d_VSRCcoeffsArray [instance_ID] [5] : 0.0 ;
                        PHASES = VSRCentry.d_VSRCfunctionOrderArray [instance_ID] > 6
                                                ? VSRCentry.d_VSRCcoeffsArray [instance_ID] [6] : 0.0 ;

                        /* compute phases in radians */
                        phasec = PHASEC * M_PI / 180.0 ;
                        phases = PHASES * M_PI / 180.0 ;
#endif

                        VA = VSRCentry.d_VSRCcoeffsArray [instance_ID] [0] ;
                        VO = VSRCentry.d_VSRCcoeffsArray [instance_ID] [1] ;
                        MF = VSRCentry.d_VSRCfunctionOrderArray [instance_ID] > 2
                                               && VSRCentry.d_VSRCcoeffsArray [instance_ID] [2]
                                               ? VSRCentry.d_VSRCcoeffsArray [instance_ID] [2] : (1 / CKTfinalTime) ;
                        FC = VSRCentry.d_VSRCfunctionOrderArray [instance_ID] > 3
                                               ? VSRCentry.d_VSRCcoeffsArray [3] [instance_ID] : 0.0 ;
                        TD = VSRCentry.d_VSRCfunctionOrderArray [instance_ID] > 4
                                               && VSRCentry.d_VSRCcoeffsArray [instance_ID] [4]
                                               ? VSRCentry.d_VSRCcoeffsArray [instance_ID] [4] : 0.0 ;

                        time -= TD ;
                        if (time <= 0)
                            value = 0 ;
                        else {

#ifdef XSPICE
                            /* compute waveform value */
                            value = VA * (VO + sin (2.0 * M_PI * MF * time + phases )) *
                                    sin (2.0 * M_PI * FC * time + phases) ;

#else
                            value = VA * (VO + sin (2.0 * M_PI * MF * time)) *
                                    sin (2.0 * M_PI * FC * time) ;
/* gtri - end - wbk - add PHASE parameters */
#endif

                        }
                    }
                    break ;

                    case PWL:
                    {
                        int i = 0, num_repeat = 0, ii = 0 ;
                        double repeat_time = 0.0, end_time, breakpt_time, itime ;

                        time -= VSRCentry.d_VSRCrdelayArray [instance_ID] ;

                        if (time < VSRCentry.d_VSRCcoeffsArray [instance_ID] [0])
                        {
                            value = VSRCentry.d_VSRCcoeffsArray [instance_ID] [1] ;
                            goto loadDone ;
                        }

                        do
                        {
                            for (i = ii ; i < (VSRCentry.d_VSRCfunctionOrderArray [instance_ID] / 2) - 1 ; i++)
                            {
                                itime = VSRCentry.d_VSRCcoeffsArray [instance_ID] [2 * i] ;
                                if (AlmostEqualUlps (itime + repeat_time, time, 3))
                                {
                                    value = VSRCentry.d_VSRCcoeffsArray [instance_ID] [2 * i + 1] ;
                                    goto loadDone ;
                                } else if ((VSRCentry.d_VSRCcoeffsArray [instance_ID] [2 * i] + repeat_time < time)
                                           && (VSRCentry.d_VSRCcoeffsArray [instance_ID] [2 * (i + 1)] +
                                           repeat_time > time))
                                {
                                    value = VSRCentry.d_VSRCcoeffsArray [instance_ID] [2 * i + 1] +
                                            (((time - (VSRCentry.d_VSRCcoeffsArray [instance_ID] [2 * i] + repeat_time)) /
                                            (VSRCentry.d_VSRCcoeffsArray [instance_ID] [2 * (i + 1)] -
                                             VSRCentry.d_VSRCcoeffsArray [instance_ID] [2 * i])) *
                                            (VSRCentry.d_VSRCcoeffsArray [instance_ID] [2 * i + 3] -
                                             VSRCentry.d_VSRCcoeffsArray [instance_ID] [2 * i + 1])) ;
                                    goto loadDone ;
                                }
                            }
                            value = VSRCentry.d_VSRCcoeffsArray [instance_ID]
                                              [VSRCentry.d_VSRCfunctionOrderArray [instance_ID] - 1] ;

                            if (!VSRCentry.d_VSRCrGivenArray [instance_ID])
                                goto loadDone ;

                            end_time = VSRCentry.d_VSRCcoeffsArray [instance_ID]
                                                 [VSRCentry.d_VSRCfunctionOrderArray [instance_ID] - 2] ;
                            breakpt_time = VSRCentry.d_VSRCcoeffsArray [instance_ID]
                                                     [VSRCentry.d_VSRCrBreakptArray [instance_ID]] ;
                            repeat_time = end_time + (end_time - breakpt_time) * (num_repeat ++) - breakpt_time ;
                            ii = VSRCentry.d_VSRCrBreakptArray [instance_ID] / 2 ;
                        } while (VSRCentry.d_VSRCrGivenArray [instance_ID]) ;
                        break ;
                    }
                } // switch
            } // else (line 55)
loadDone:

#ifdef XSPICE_EXP
/* gtri - begin - wbk - modify for supply ramping option */
            value *= CKTsrcFact ;
            value *= cm_analog_ramp_factor () ;
#else
            if (CKTmode & MODETRANOP)
                value *= CKTsrcFact ;
/* gtri - end - wbk - modify to process srcFact, etc. for all sources */
#endif

            d_CKTloadOutputRHS [d_PositionVectorRHS [instance_ID]] = value ;
        }
    }

    return ;
}

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
#include "isrcdefs.h"

#ifdef XSPICE_EXP
/* gtri - begin - wbk - modify for supply ramping option */
#include "ngspice/cmproto.h"
/* gtri - end   - wbk - modify for supply ramping option */
#endif



/*** TRNOISE and TRRANDOM don't work in the CUDA implementation ***/

/* cudaMemcpy MACRO to check it for errors --> CUDAMEMCPYCHECK(name of pointer, dimension, type, status) */
#define CUDAMEMCPYCHECK(a, b, c, d) \
    if (d != cudaSuccess) \
    { \
        fprintf (stderr, "cuISRCload routine...\n") ; \
        fprintf (stderr, "Error: cudaMemcpy failed on %s size of %d bytes\n", #a, (int)(b * sizeof(c))) ; \
        fprintf (stderr, "Error: %s = %d, %s\n", #d, d, cudaGetErrorString (d)) ; \
        return (E_NOMEM) ; \
    }

extern "C"
__global__ void cuISRCload_kernel (ISRCparamGPUstruct, int, double, double, double, double, int, int *, double *) ;

extern "C"
int
cuISRCload
(
GENmodel *inModel, CKTcircuit *ckt
)
{
    ISRCmodel *model = (ISRCmodel *)inModel ;
    int thread_x, thread_y, block_x ;

    cudaError_t status ;

    /*  loop through all the inductor models */
    for ( ; model != NULL ; model = ISRCnextModel(model))
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

        cuISRCload_kernel <<< block_x, thread >>> (model->ISRCparamGPU, ckt->CKTmode, ckt->CKTtime,
                                                   ckt->CKTstep, ckt->CKTfinalTime, ckt->CKTsrcFact,
                                                   model->n_instances, model->d_PositionVectorRHS,
                                                   ckt->d_CKTloadOutputRHS) ;

        cudaDeviceSynchronize () ;

        status = cudaGetLastError () ; // check for launch error
        if (status != cudaSuccess)
        {
            fprintf (stderr, "Kernel launch failure in the Current Source Model\n\n") ;
            return (E_NOMEM) ;
        }
    }

    return (OK) ;
}

extern "C"
__global__
void
cuISRCload_kernel
(
ISRCparamGPUstruct ISRCentry, int CKTmode, double CKTtime,
double CKTstep, double CKTfinalTime, double CKTsrcFact, int n_instances,
int *d_PositionVectorRHS, double *d_CKTloadOutputRHS
)
{
    int instance_ID ;
    double value, time ;

    instance_ID = threadIdx.y + blockDim.y * blockIdx.x ;

    if (instance_ID < n_instances)
    {
        if (threadIdx.x == 0)
        {
            if ((CKTmode & (MODEDCOP | MODEDCTRANCURVE)) && ISRCentry.d_ISRCdcGivenArray [instance_ID])
            {
                /* load using DC value */

#ifdef XSPICE_EXP
/* gtri - begin - wbk - modify to process srcFact, etc. for all sources */
                value = ISRCentry.d_ISRCdcvalueArray [instance_ID] ;
#else
                value = ISRCentry.d_ISRCdcvalueArray [instance_ID] * CKTsrcFact ;
#endif

            } else {
                if (CKTmode & (MODEDC))
                    time = 0 ;
                else
                    time = CKTtime ;

                /* use the transient functions */
                switch (ISRCentry.d_ISRCfunctionTypeArray [instance_ID])
                {
                    default:

#ifdef XSPICE_EXP
                        value = ISRCentry.d_ISRCdcvalueArray [instance_ID] ;
#else
                        value = ISRCentry.d_ISRCdcvalueArray [instance_ID] * CKTsrcFact ;
#endif

                        break ;

                    case PULSE:
                    {
                        double V1, V2, TD, TR, TF, PW, PER, basetime = 0 ;

#ifdef XSPICE
                        double PHASE, phase, deltat ;
#endif

                        V1 = ISRCentry.d_ISRCcoeffsArray [instance_ID] [0] ;
                        V2 = ISRCentry.d_ISRCcoeffsArray [instance_ID] [1] ;
                        TD = ISRCentry.d_ISRCfunctionOrderArray [instance_ID] > 2
                                              ? ISRCentry.d_ISRCcoeffsArray [instance_ID] [2] : 0.0 ;
                        TR = ISRCentry.d_ISRCfunctionOrderArray [instance_ID] > 3
                                              && ISRCentry.d_ISRCcoeffsArray [instance_ID] [3] != 0.0
                                              ? ISRCentry.d_ISRCcoeffsArray [instance_ID] [3] : CKTstep ;
                        TF = ISRCentry.d_ISRCfunctionOrderArray [instance_ID] > 4
                                              && ISRCentry.d_ISRCcoeffsArray [instance_ID] [4] != 0.0
                                              ? ISRCentry.d_ISRCcoeffsArray [instance_ID] [4] : CKTstep ;
                        PW = ISRCentry.d_ISRCfunctionOrderArray [instance_ID] > 5
                                              && ISRCentry.d_ISRCcoeffsArray [instance_ID] [5] != 0.0
                                              ? ISRCentry.d_ISRCcoeffsArray [instance_ID] [5] : CKTfinalTime ;
                        PER = ISRCentry.d_ISRCfunctionOrderArray [instance_ID] > 6
                                              && ISRCentry.d_ISRCcoeffsArray [instance_ID] [6] != 0.0
                                              ? ISRCentry.d_ISRCcoeffsArray [instance_ID] [6] : CKTfinalTime ;

                        /* shift time by delay time TD */
                        time -=  TD ;

#ifdef XSPICE
/* gtri - begin - wbk - add PHASE parameter */
                        PHASE = ISRCentry.d_ISRCfunctionOrderArray [instance_ID] > 7
                                              ? ISRCentry.d_ISRCcoeffsArray [instance_ID] [7] : 0.0 ;

                        /* normalize phase to cycles */
                        phase = PHASE / 360.0 ;
                        phase = fmod (phase, 1.0) ;
                        deltat =  phase * PER ;
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
                            value = V1 + (V2 - V1) * (time) / TR ;
                        else /* time > TR + PW && < TR + PW + TF */
                            value = V2 + (V1 - V2) * (time - (TR + PW)) / TF ;
                    }
                    break ;

                    case SINE:
                    {
                        double VO, VA, FREQ, TD, THETA ;

/* gtri - begin - wbk - add PHASE parameter */
#ifdef XSPICE
                        double PHASE, phase ;

                        PHASE = ISRCentry.d_ISRCfunctionOrderArray [instance_ID] > 5
                                              ? ISRCentry.d_ISRCcoeffsArray [instance_ID] [5] : 0.0 ;

                        /* compute phase in radians */
                        phase = PHASE * M_PI / 180.0 ;
#endif

                        VO = ISRCentry.d_ISRCcoeffsArray [instance_ID] [0] ;
                        VA = ISRCentry.d_ISRCcoeffsArray [instance_ID] [1] ;
                        FREQ = ISRCentry.d_ISRCfunctionOrderArray [instance_ID] > 2
                                               && ISRCentry.d_ISRCcoeffsArray [instance_ID] [2] != 0.0
                                               ? ISRCentry.d_ISRCcoeffsArray [instance_ID] [2] : (1 / CKTfinalTime) ;
                        TD = ISRCentry.d_ISRCfunctionOrderArray [instance_ID] > 3
                                               ? ISRCentry.d_ISRCcoeffsArray [instance_ID] [3] : 0.0 ;
                        THETA = ISRCentry.d_ISRCfunctionOrderArray [instance_ID] > 4
                                               ? ISRCentry.d_ISRCcoeffsArray [instance_ID] [4] : 0.0 ;

                        time -= TD ;
                        if (time <= 0)

#ifdef XSPICE
                            value = VO + VA * sin (phase) ;
                        else
                            value = VO + VA * sin (FREQ * time * 2.0 * M_PI + phase) * exp (-time * THETA) ;
#else
                            value = VO ;
                        else
                            value = VO + VA * sin (FREQ * time * 2.0 * M_PI) * exp (-time * THETA) ;
#endif
/* gtri - end - wbk - add PHASE parameter */

                    }
                    break ;

                    case EXP:
                    {
                        double V1, V2, TD1, TD2, TAU1, TAU2 ;

                        V1 = ISRCentry.d_ISRCcoeffsArray [instance_ID] [0] ;
                        V2 = ISRCentry.d_ISRCcoeffsArray [instance_ID] [1] ;
                        TD1 = ISRCentry.d_ISRCfunctionOrderArray [instance_ID] > 2
                                              && ISRCentry.d_ISRCcoeffsArray [instance_ID] [2] != 0.0
                                              ? ISRCentry.d_ISRCcoeffsArray [instance_ID] [2] : CKTstep ;
                        TAU1 = ISRCentry.d_ISRCfunctionOrderArray [instance_ID] > 3
                                              && ISRCentry.d_ISRCcoeffsArray [instance_ID] [3] != 0.0
                                              ? ISRCentry.d_ISRCcoeffsArray [instance_ID] [3] : CKTstep ;
                        TD2 = ISRCentry.d_ISRCfunctionOrderArray [instance_ID] > 4
                                              && ISRCentry.d_ISRCcoeffsArray [instance_ID] [4] != 0.0
                                              ? ISRCentry.d_ISRCcoeffsArray [instance_ID] [4] : TD1 + CKTstep ;
                        TAU2 = ISRCentry.d_ISRCfunctionOrderArray [instance_ID] > 5
                                              && ISRCentry.d_ISRCcoeffsArray [instance_ID] [5]
                                              ? ISRCentry.d_ISRCcoeffsArray [instance_ID] [5] : CKTstep ;

                        if (time <= TD1)
                            value = V1 ;
                        else if (time <= TD2)
                            value = V1 + (V2 - V1) * (1 - exp (-(time - TD1) / TAU1)) ;
                        else
                            value = V1 + (V2 - V1) * (1 - exp (-(time - TD1) / TAU1)) +
                                         (V1 - V2) * (1 - exp (-(time - TD2) / TAU2)) ;
                    }
                    break ;

                    case SFFM:
                    {
                        double VO, VA, FC, MDI, FS ;

/* gtri - begin - wbk - add PHASE parameters */
#ifdef XSPICE
                        double PHASEC, PHASES, phasec, phases ;

                        PHASEC = ISRCentry.d_ISRCfunctionOrderArray [instance_ID] > 5
                                              ? ISRCentry.d_ISRCcoeffsArray [instance_ID] [5] : 0.0 ;
                        PHASES = ISRCentry.d_ISRCfunctionOrderArray [instance_ID] > 6
                                              ? ISRCentry.d_ISRCcoeffsArray [instance_ID] [6] : 0.0 ;

                        /* compute phases in radians */
                        phasec = PHASEC * M_PI / 180.0 ;
                        phases = PHASES * M_PI / 180.0 ;
#endif

                        VO = ISRCentry.d_ISRCcoeffsArray [instance_ID] [0] ;
                        VA = ISRCentry.d_ISRCcoeffsArray [instance_ID] [1] ;
                        FC = ISRCentry.d_ISRCfunctionOrderArray [instance_ID] > 2
                                              && ISRCentry.d_ISRCcoeffsArray [instance_ID] [2]
                                              ? ISRCentry.d_ISRCcoeffsArray [instance_ID] [2] : (1 / CKTfinalTime) ;
                        MDI = ISRCentry.d_ISRCfunctionOrderArray [instance_ID] > 3
                                              ? ISRCentry.d_ISRCcoeffsArray [instance_ID] [3] : 0.0 ;
                        FS = ISRCentry.d_ISRCfunctionOrderArray [instance_ID] > 4
                                              && ISRCentry.d_ISRCcoeffsArray [instance_ID] [4]
                                              ? ISRCentry.d_ISRCcoeffsArray [instance_ID] [4] : (1 / CKTfinalTime) ;

#ifdef XSPICE
                        /* compute waveform value */
                        value = VO + VA * sin ((2.0 * M_PI * FC * time + phasec) +
                                MDI * sin (2.0 * M_PI * FS * time + phases)) ;
#else
                        value = VO + VA * sin ((2.0 * M_PI * FC * time) +
                                MDI * sin (2.0 * M_PI * FS * time)) ;
#endif
/* gtri - end - wbk - add PHASE parameters */

                    }
                    break ;

                    case AM:
                    {
                        double VA, FC, MF, VO, TD ;

/* gtri - begin - wbk - add PHASE parameters */
#ifdef XSPICE
                        double PHASEC, PHASES, phasec, phases ;

                        PHASEC = ISRCentry.d_ISRCfunctionOrderArray [instance_ID] > 5
                                              ? ISRCentry.d_ISRCcoeffsArray [instance_ID] [5] : 0.0 ;
                        PHASES = ISRCentry.d_ISRCfunctionOrderArray [instance_ID] > 6
                                              ? ISRCentry.d_ISRCcoeffsArray [instance_ID] [6] : 0.0 ;

                        /* compute phases in radians */
                        phasec = PHASEC * M_PI / 180.0 ;
                        phases = PHASES * M_PI / 180.0 ;
#endif

                        VA = ISRCentry.d_ISRCcoeffsArray [instance_ID] [0] ;
                        VO = ISRCentry.d_ISRCcoeffsArray [instance_ID] [1] ;
                        MF = ISRCentry.d_ISRCfunctionOrderArray [instance_ID] > 2
                                              && ISRCentry.d_ISRCcoeffsArray [instance_ID] [2]
                                              ? ISRCentry.d_ISRCcoeffsArray [instance_ID] [2] : (1 / CKTfinalTime) ;
                        FC = ISRCentry.d_ISRCfunctionOrderArray [instance_ID] > 3
                                              ? ISRCentry.d_ISRCcoeffsArray [instance_ID] [3] : 0.0 ;
                        TD = ISRCentry.d_ISRCfunctionOrderArray [instance_ID] > 4
                                              && ISRCentry.d_ISRCcoeffsArray [instance_ID] [4]
                                              ? ISRCentry.d_ISRCcoeffsArray [instance_ID] [4] : 0.0 ;

                        time -= TD ;
                        if (time <= 0)
                            value = 0 ;
                        else
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
                    break ;

                    case PWL:
                    {
                        int i ;
                        if (time < ISRCentry.d_ISRCcoeffsArray [instance_ID] [0])
                        {
                            value = ISRCentry.d_ISRCcoeffsArray [instance_ID] [1] ;
                            break ;
                        }

                        for (i = 0 ; i <= (ISRCentry.d_ISRCfunctionOrderArray [instance_ID] / 2) - 1 ; i++)
                        {
                            if ((ISRCentry.d_ISRCcoeffsArray [instance_ID] [2 * i] == time))
                            {
                                value = ISRCentry.d_ISRCcoeffsArray [instance_ID] [2 * i + 1] ;
                                goto loadDone ;
                            }
                            if ((ISRCentry.d_ISRCcoeffsArray [instance_ID] [2 * i] < time)
                                 && (ISRCentry.d_ISRCcoeffsArray [instance_ID] [2 * (i + 1)] > time))
                            {
                                value = ISRCentry.d_ISRCcoeffsArray [instance_ID] [2 * i + 1] +
                                       (((time - ISRCentry.d_ISRCcoeffsArray [instance_ID] [2 * i]) /
                                       (ISRCentry.d_ISRCcoeffsArray [instance_ID] [2 * (i + 1)] -
                                        ISRCentry.d_ISRCcoeffsArray [instance_ID] [2 * i])) *
                                       (ISRCentry.d_ISRCcoeffsArray [instance_ID] [2 * i + 3] -
                                        ISRCentry.d_ISRCcoeffsArray [instance_ID] [2 * i + 1])) ;
                                goto loadDone ;
                            }
                        }
                        value = ISRCentry.d_ISRCcoeffsArray [instance_ID]
                                [ISRCentry.d_ISRCfunctionOrderArray [instance_ID] - 1] ;
                        break ;
                    }
                } // switch
            } // else (line 593)

loadDone:

#ifdef XSPICE_EXP
/* gtri - begin - wbk - modify for supply ramping option */
            value *= CKTsrcFact ;
            value *= cm_analog_ramp_factor () ;
#else
            if (CKTmode & MODETRANOP)
                value *= CKTsrcFact ;
/* gtri - end - wbk - modify for supply ramping option */
#endif

            d_CKTloadOutputRHS [d_PositionVectorRHS [instance_ID]] = value ;

/* gtri - end - wbk - modify to process srcFact, etc. for all sources */

#ifdef XSPICE
/* gtri - begin - wbk - record value so it can be output if requested */
            here->ISRCcurrent = value ;
/* gtri - end   - wbk - record value so it can be output if requested */
#endif
        }
    }

    return ;
}

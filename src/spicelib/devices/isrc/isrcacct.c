/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "isrcdefs.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/missing_math.h"
#include "ngspice/1-f-code.h"
#include "ngspice/compatmode.h"

#ifndef HAVE_LIBFFTW3
extern void fftFree(void);
#endif

extern bool ft_ngdebug; /* some additional debug info printed */

#define SAMETIME(a,b)    (fabs((a)-(b))<= TIMETOL * PW)
#define TIMETOL    1e-7

int
ISRCaccept(CKTcircuit *ckt, GENmodel *inModel)
        /* set up the breakpoint table.  */
{
    ISRCmodel *model = (ISRCmodel *) inModel;
    ISRCinstance *here;
    int error;

    /*  loop through all the voltage source models */
    for( ; model != NULL; model = ISRCnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = ISRCinstances(model); here != NULL ;
                here=ISRCnextInstance(here)) {

            if(!(ckt->CKTmode & (MODETRAN | MODETRANOP))) {
                /* not transient, so shouldn't be here */
                return(OK);
            } else {
                /* use the transient functions */
                switch(here->ISRCfunctionType) {

                    default: { /* no function specified:DC   no breakpoints */
                        break;
                    }

                    case PULSE: {

                        double TD, TR, TF, PW, PER;
                        double tshift;
                        double time = 0.;
                        double basetime = 0;
                        double PHASE;
                        double phase;
                        double deltat;
                        double tmax = 1e99;

                        TD = here->ISRCfunctionOrder > 2
                            ? here->ISRCcoeffs[2] : 0.0;
                        TR = here->ISRCfunctionOrder > 3
                            && here->ISRCcoeffs[3] != 0.0
                            ? here->ISRCcoeffs[3] : ckt->CKTstep;
                        TF = here->ISRCfunctionOrder > 4
                            && here->ISRCcoeffs[4] != 0.0
                            ? here->ISRCcoeffs[4] : ckt->CKTstep;
                        PW = here->ISRCfunctionOrder > 5
                            && here->ISRCcoeffs[5] != 0.0
                            ? here->ISRCcoeffs[5] : ckt->CKTfinalTime;
                        PER = here->ISRCfunctionOrder > 6
                            && here->ISRCcoeffs[6] != 0.0
                            ? here->ISRCcoeffs[6] : ckt->CKTfinalTime;
                        PHASE = here->ISRCfunctionOrder > 7
                            ? here->ISRCcoeffs[7] : 0.0;

                        /* offset time by delay */
                        time = ckt->CKTtime - TD;
                        tshift = TD;

                        if (newcompat.xs) {
                         /* normalize phase to 0 - 360° */
                         /* normalize phase to cycles */
                            phase = PHASE / 360.0;
                            phase = fmod(phase, 1.0);
                            deltat =  phase * PER;
                            while (deltat > 0)
                                deltat -= PER;
                            time += deltat;
                            tshift = TD - deltat;
                        }
                        else if (PHASE > 0.0) {
                            tmax = PHASE * PER;
                        }

                        if (!newcompat.xs && time > tmax) {
                            /* Do nothing */
                        }
                        else {
                            if (time >= PER) {
                                /* repeating signal - figure out where we are */
                                /* in period */
                                basetime = PER * floor(time / PER);
                                time -= basetime;
                            }

                            if (time <= 0.0 || time >= TR + PW + TF) {
                                if (ckt->CKTbreak && SAMETIME(time, 0.0)) {
                                    error = CKTsetBreak(ckt, basetime + TR + tshift);
                                    if (error) return(error);
                                }
                                else if (ckt->CKTbreak && SAMETIME(TR + PW + TF, time)) {
                                    error = CKTsetBreak(ckt, basetime + PER + tshift);
                                    if (error) return(error);
                                }
                                else if (ckt->CKTbreak && (time == -tshift)) {
                                    error = CKTsetBreak(ckt, basetime + tshift);
                                    if (error) return(error);
                                }
                                else if (ckt->CKTbreak && SAMETIME(PER, time)) {
                                    error = CKTsetBreak(ckt, basetime + tshift + TR + PER);
                                    if (error) return(error);
                                }
                            }
                            else  if (time >= TR && time <= TR + PW) {
                                if (ckt->CKTbreak && SAMETIME(time, TR)) {
                                    error = CKTsetBreak(ckt, basetime + tshift + TR + PW);
                                    if (error) return(error);
                                }
                                else if (ckt->CKTbreak && SAMETIME(TR + PW, time)) {
                                    error = CKTsetBreak(ckt, basetime + tshift + TR + PW + TF);
                                    if (error) return(error);
                                }
                            }
                            else if (time > 0 && time < TR) {
                                if (ckt->CKTbreak && SAMETIME(time, 0)) {
                                    error = CKTsetBreak(ckt, basetime + tshift + TR);
                                    if (error) return(error);
                                }
                                else if (ckt->CKTbreak && SAMETIME(time, TR)) {
                                    error = CKTsetBreak(ckt, basetime + tshift + TR + PW);
                                    if (error) return(error);
                                }
                            }
                            else { /* time > TR + PW && < TR + PW + TF */
                                if (ckt->CKTbreak && SAMETIME(time, TR + PW)) {
                                    error = CKTsetBreak(ckt, basetime + tshift + TR + PW + TF);
                                    if (error) return(error);
                                }
                                else if (ckt->CKTbreak && SAMETIME(time, TR + PW + TF)) {
                                    error = CKTsetBreak(ckt, basetime + tshift + PER);
                                    if (error) return(error);
                                }
                            }
                        }
                    }
                    break;

                    case SINE: {
                        /* no  breakpoints (yet) */
                    }
                    break;

                    case EXP: {
                        /* no  breakpoints (yet) */
                    }
                    break;

                    case SFFM:{
                        /* no  breakpoints (yet) */
                    }
                    break;

                    case AM:{
                        /* no  breakpoints (yet) */
                    }
                    break;

                    case PWL: {
                        int i;
                        if(ckt->CKTtime < *(here->ISRCcoeffs)) {
                            if(ckt->CKTbreak) {
                                error = CKTsetBreak(ckt,*(here->ISRCcoeffs));
                                break;
                            }
                        }
                        for(i=0;i<(here->ISRCfunctionOrder/2)-1;i++) {
                            if ( ckt->CKTbreak && AlmostEqualUlps(*(here->ISRCcoeffs+2*i), ckt->CKTtime, 3 ) ) {
                                error = CKTsetBreak(ckt, *(here->ISRCcoeffs+2*i+2));
                                if(error) return(error);
                                goto bkptset;
                            }
                        }
                        break;
                    }

    /**** tansient noise routines:
    INoi2 2 0  DC 0 TRNOISE(10n 0.5n 0 0n) : generate gaussian distributed noise
                            rms value, time step, 0 0
    INoi1 1 0  DC 0 TRNOISE(0n 0.5n 1 10n) : generate 1/f noise
                            0,  time step, exponent < 2, rms value
    */
                    case TRNOISE: {

                        struct trnoise_state *state = here -> ISRCtrnoise_state;
                        double TS = state -> TS;
                        double RTSAM = state ->RTSAM;

                        if ((TS == 0.0) &&  (RTSAM == 0.0)) // no further breakpoint if value not given
                            break;

#ifndef HAVE_LIBFFTW3
                        /* FIXME, dont' want this here, over to aof_get or somesuch */
                        if (ckt->CKTtime == 0.0) {
                            if (ft_ngdebug)
                                printf("ISRC: free fft tables\n");
                            fftFree();
                        }
#endif

                        if(ckt->CKTbreak) {

                            int n = (int) floor(ckt->CKTtime / TS + 0.5);
                            volatile double nearest = n * TS;

                            if(AlmostEqualUlps(nearest, ckt->CKTtime, 3)) {
                                /* carefull calculate `next'
                                *  make sure it is really identical
                                *  with the next calculated `nearest' value
                                */
                                volatile double next = (n+1) * TS;
                                error = CKTsetBreak(ckt, next);
                                if(error)
                                    return(error);
                            }
                        }

                        if (RTSAM > 0) {
                            double RTScapTime = state->RTScapTime;
                            double RTSemTime = state->RTSemTime;
                            double RTSCAPT = state->RTSCAPT;
                            double RTSEMT = state->RTSEMT;

                            if (ckt->CKTtime == 0) {
                                /* initialzing here again needed for repeated calls to tran command */
                                state->RTScapTime = RTScapTime = exprand(RTSCAPT);
                                state->RTSemTime = RTSemTime = RTScapTime + exprand(RTSEMT);
                                if (ckt->CKTbreak) {
                                    error = CKTsetBreak(ckt, RTScapTime);
                                    if(error)
                                        return(error);
                                }
                            }

                            if(AlmostEqualUlps(RTScapTime, ckt->CKTtime, 3)) {
                                if (ckt->CKTbreak) {
                                    error = CKTsetBreak(ckt, RTSemTime);
                                    if(error)
                                        return(error);
                                }
                            }

                            if(AlmostEqualUlps(RTSemTime, ckt->CKTtime, 3)) {
                                /* new values */
                                RTScapTime = here -> ISRCtrnoise_state ->RTScapTime = ckt->CKTtime + exprand(RTSCAPT);
                                here -> ISRCtrnoise_state ->RTSemTime = RTScapTime + exprand(RTSEMT);

                                if (ckt->CKTbreak) {
                                    error = CKTsetBreak(ckt, RTScapTime);
                                    if(error)
                                        return(error);
                                }
                            }
                        }
                    }
                    break;

                    case TRRANDOM: {
                        struct trrandom_state *state = here -> ISRCtrrandom_state;
                        double TS = state -> TS;
                        double TD = state -> TD;

                        double time = ckt->CKTtime - TD;

                        if (time < 0) break;

                        if(ckt->CKTbreak) {

                            int n = (int) floor(time / TS + 0.5);
                            volatile double nearest = n * TS;

                            if(AlmostEqualUlps(nearest, time, 3)) {
                            /* carefully calculate `next'
                            *  make sure it is really identical
                            *  with the next calculated `nearest' value
                            */
                                volatile double next = (n+1) * TS + TD;
                                error = CKTsetBreak(ckt, next);
                                if(error)
                                    return(error);
                                state->value = trrandom_state_get(state);
                            }
                        }
                    }
                    break;

#ifdef SHARED_MODULE
                    case EXTERNAL: {
                        /* no  breakpoints (yet) */
                    }
                    break;
#endif

                } // switch
            } // if ... else
bkptset: ;
        } // for
    } // for

    return(OK);
}

/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vsrcdefs.h"
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

int
VSRCaccept(CKTcircuit *ckt, GENmodel *inModel)
        /* set up the breakpoint table.  */
{
    VSRCmodel *model = (VSRCmodel *) inModel;
    VSRCinstance *here;
    int error;

    /*  loop through all the voltage source models */
    for( ; model != NULL; model = VSRCnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = VSRCinstances(model); here != NULL ;
                here=VSRCnextInstance(here)) {

            if(!(ckt->CKTmode & (MODETRAN | MODETRANOP))) {
                /* not transient, so shouldn't be here */
                return(OK);
            } else {
                /* use the transient functions */
                switch(here->VSRCfunctionType) {

                    default: { /* no function specified:DC   no breakpoints */
                        break;
                    }

                    case PULSE: {

                        double TD, TR, TF, PW, PER;
                        double time = 0.;
                        double basetime = 0;
                        double tmax = 1e99;

                        double PHASE;
                        double phase;
                        double deltat;

                        TD = here->VSRCfunctionOrder > 2
                            ? here->VSRCcoeffs[2] : 0.0;
                        TR = here->VSRCfunctionOrder > 3
                            && here->VSRCcoeffs[3] != 0.0
                            ? here->VSRCcoeffs[3] : ckt->CKTstep;
                        TF = here->VSRCfunctionOrder > 4
                            && here->VSRCcoeffs[4] != 0.0
                            ? here->VSRCcoeffs[4] : ckt->CKTstep;
                        PW = here->VSRCfunctionOrder > 5
                            && here->VSRCcoeffs[5] != 0.0
                            ? here->VSRCcoeffs[5] : ckt->CKTfinalTime;
                        PER = here->VSRCfunctionOrder > 6
                            && here->VSRCcoeffs[6] != 0.0
                            ? here->VSRCcoeffs[6] : ckt->CKTfinalTime;
                        PHASE = here->VSRCfunctionOrder > 7
                            ? here->VSRCcoeffs[7] : 0.0;

                        /* offset time by delay */
                        time = ckt->CKTtime - TD;

                        if (newcompat.xs) {
                            /* normalize phase to 0 - 360° */
                            /* normalize phase to cycles */
                            phase = PHASE / 360.0;
                            phase = fmod(phase, 1.0);
                            deltat = phase * PER;
                            while (deltat > 0)
                                deltat -= PER;
                            time += deltat;
                        } else if (PHASE > 0.0) {
                            tmax = PHASE * PER;
                            if (time > tmax)
                                break;
                        }

                        if (ckt->CKTtime >= here->VSRCbreak_time) {
                            double wait;

                            if (time >= PER) {
                                /* Repeating signal: where in period are we? */

                                basetime = PER * floor(time / PER);
                                time -= basetime;
                            }

                            /* Set next breakpoint. */

                            if (time < 0.0) {
                                /* Await first pulse */

                                wait = -time;
                            } else if (time < TR) {
                                /* Wait for end of rise. */

                                wait = TR - time;
                            } else if (time < TR + PW) {
                                /* Wait for fall. */

                                wait = TR + PW - time;
                            } else if (time < TR + PW + TF) {
                                /* Wait for end of fall. */

                                wait = TR + PW + TF - time;
                            } else {
                                /* Wait for next pulse. */
                                wait = PER - time;
                            }
                            here->VSRCbreak_time = ckt->CKTtime + wait;
                            error = CKTsetBreak(ckt, here->VSRCbreak_time);
                            if (error)
                                return error;

                            /* If a timestep ends just before the break time,
                             * the break request may be ignored.
                             * Set threshold for requesting following break.
                             */

                            here->VSRCbreak_time -= ckt->CKTminBreak;
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

                    case PWL:
                        if (ckt->CKTtime >= here->VSRCbreak_time) {
                            double time, end, period;
                            int    i;

                            time = ckt->CKTtime - here->VSRCrdelay;
                            end =
                                here->VSRCcoeffs[here->VSRCfunctionOrder - 2];
                            if (time > end) {
                                if (here->VSRCrGiven) {
                                    /* Repeating. */

                                    period = end -
                                        here->VSRCcoeffs[here->VSRCrBreakpt];
                                    time -=
                                        here->VSRCcoeffs[here->VSRCrBreakpt];
                                    time -= period * floor(time / period);
                                    time +=
                                        here->VSRCcoeffs[here->VSRCrBreakpt];
                                } else {
                                     here->VSRCbreak_time = ckt->CKTfinalTime;
                                     break;
                                }
                            }

                            for (i = 0;
                                 i < here->VSRCfunctionOrder;
                                 i += 2) {
                                if (here->VSRCcoeffs[i] > time) {
                                    here->VSRCbreak_time =
                                        ckt->CKTtime +
                                            here->VSRCcoeffs[i] - time;
                                    error = CKTsetBreak(ckt,
                                                        here->VSRCbreak_time);
                                    if (error)
                                        return error;
                                    here->VSRCbreak_time -= ckt->CKTminBreak;
                                    break;
                                }
                            }
                        }
                        break;

    /**** transient noise routines:
    VNoi2 2 0  DC 0 TRNOISE(10n 0.5n 0 0n) : generate gaussian distributed noise
                            rms value, time step, 0 0
    VNoi1 1 0  DC 0 TRNOISE(0n 0.5n 1 10n) : generate 1/f noise
                            0,  time step, exponent < 2, rms value
    */
                    case TRNOISE: {


                        struct trnoise_state *state = here -> VSRCtrnoise_state;
                        double TS = state -> TS;
                        double RTSAM = state ->RTSAM;

                        if ((TS == 0.0) &&  (RTSAM == 0.0)) // no further breakpoint if value not given
                            break;

#ifndef HAVE_LIBFFTW3
                        /* FIXME, dont' want this here, over to aof_get or somesuch */
                        if (ckt->CKTtime == 0.0) {
                            if(ft_ngdebug)
                                printf("VSRC: free fft tables\n");
                            fftFree();
                        }
#endif
                        if (TS > 0 && ckt->CKTtime >= here->VSRCbreak_time) {
                            if (here->VSRCbreak_time < 0.0)
                                here->VSRCbreak_time = TS;
                            else
                                here->VSRCbreak_time += TS;
                            error = CKTsetBreak(ckt, here->VSRCbreak_time);
                            if (error)
                                return(error);
                            here->VSRCbreak_time -= ckt->CKTminBreak;
                        }

                        if (RTSAM <= 0)
                            break;      /* No shot noise. */

                        if (ckt->CKTtime == 0) {
                            /* initialzing here again needed for repeated calls to tran command */
                            state->RTScapTime = exprand(state->RTSCAPT);
                            state->RTSemTime =
                                state->RTScapTime + exprand(state->RTSEMT);
                            error = CKTsetBreak(ckt, state->RTScapTime);
                            if(error)
                                return(error);
                            break;
                        }

                        /* Break handling code ends a timestep close to
                         * the requested time.
                         */

                        if (ckt->CKTtime >=
                                state->RTScapTime - ckt->CKTminBreak &&
                            ckt->CKTtime <=
                            state->RTScapTime + ckt->CKTminBreak) {
                            error = CKTsetBreak(ckt, state->RTSemTime);
                            if(error)
                                return(error);
                        }

                        if (ckt->CKTtime >=
                                state->RTSemTime - ckt->CKTminBreak) {
                            /* new values */

                            state->RTScapTime =
                                ckt->CKTtime + exprand(state->RTSCAPT);
                            state->RTSemTime =
                                state->RTScapTime + exprand(state->RTSEMT);
                            error = CKTsetBreak(ckt, state->RTScapTime);
                            if(error)
                                return(error);
                        }
                    }
                    break;

                    case TRRANDOM: {
                        struct trrandom_state *state = here -> VSRCtrrandom_state;
                        double TS = state -> TS;
                        double TD = state -> TD;

                        if (ckt->CKTtime == 0 && TD > 0) {
                            error = CKTsetBreak(ckt, TD);
                            here->VSRCbreak_time = TD;
                            if (error)
                                return(error);
                            break;
                        }

                        if (ckt->CKTtime >= here->VSRCbreak_time) {
                            if (here->VSRCbreak_time < 0.0)
                                here->VSRCbreak_time = TS;
                            else
                                here->VSRCbreak_time += TS;
                            error = CKTsetBreak(ckt, here->VSRCbreak_time);
                            if (error)
                                return(error);
                            here->VSRCbreak_time -= ckt->CKTminBreak;
                            state->value = trrandom_state_get(state);
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
        } // for
    } // for

    return(OK);
}

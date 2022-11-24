/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 Alansfixes
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "isrcdefs.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/1-f-code.h"
#include "ngspice/compatmode.h"

#ifdef XSPICE_EXP
/* gtri - begin - wbk - modify for supply ramping option */
#include "ngspice/cmproto.h"
/* gtri - end   - wbk - modify for supply ramping option */
#endif

#ifdef SHARED_MODULE
extern double getisrcval(double, char*);
#endif

int
ISRCload(GENmodel *inModel, CKTcircuit *ckt)
        /* actually load the current value into the
         * sparse matrix previously provided
         */
{
    ISRCmodel *model = (ISRCmodel *) inModel;
    ISRCinstance *here;
    double value;
    double time;
    double m;

    /*  loop through all the source models */
    for( ; model != NULL; model = ISRCnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = ISRCinstances(model); here != NULL ;
                here=ISRCnextInstance(here)) {

            m = here->ISRCmValue;

            if( (ckt->CKTmode & (MODEDCOP | MODEDCTRANCURVE)) &&
                    here->ISRCdcGiven ) {
                /* load using DC value */
#ifdef XSPICE_EXP
/* gtri - begin - wbk - modify to process srcFact, etc. for all sources */
                value = here->ISRCdcValue;
#else
                value = here->ISRCdcValue * ckt->CKTsrcFact;
#endif
            } else {
                if(ckt->CKTmode & (MODEDC)) {
                    time = 0;
                } else {
                    time = ckt->CKTtime;
                }
                /* use the transient functions */
                switch(here->ISRCfunctionType) {

                    default:
#ifdef XSPICE_EXP
                        value = here->ISRCdcValue;
#else
                        value = here->ISRCdcValue * ckt->CKTsrcFact;
#endif
                        break;

                    case PULSE: {
                        double V1, V2, TD, TR, TF, PW, PER;
                        double basetime = 0;
                        double PHASE;
                        double phase;
                        double deltat;
                        double tmax = 1e99;

                        V1 = here->ISRCcoeffs[0];
                        V2 = here->ISRCcoeffs[1];
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

                        /* shift time by delay time TD */
                        time -=  TD;

                        PHASE = here->ISRCfunctionOrder > 7
                           ? here->ISRCcoeffs[7] : 0.0;

                        if (newcompat.xs) { /* 7th parameter is PHASE */
                            /* normalize phase to cycles */
                            phase = PHASE / 360.0;
                            phase = fmod(phase, 1.0);
                            deltat =  phase * PER;
                            while (deltat > 0)
                                deltat -= PER;
                            /* shift time by pase (neg. for pos. phase value) */
                            time += deltat;
                        }
                        else if (PHASE > 0.0) { /* 7th parameter is number of pulses */
                            tmax = PHASE * PER;
                        }

                        if (!newcompat.xs && time > tmax) {
                            value = V1;
                        }
                        else {
                            if (time > PER) {
                                /* repeating signal - figure out where we are */
                                /* in period */
                                basetime = PER * floor(time / PER);
                                time -= basetime;
                            }
                            if (time <= 0 || time >= TR + PW + TF) {
                                value = V1;
                            }
                            else  if (time >= TR && time <= TR + PW) {
                                value = V2;
                            }
                            else if (time > 0 && time < TR) {
                                value = V1 + (V2 - V1) * (time) / TR;
                            }
                            else { /* time > TR + PW && < TR + PW + TF */
                                value = V2 + (V1 - V2) * (time - (TR + PW)) / TF;
                            }
                        }
                    }
                    break;

                    case SINE: {

                        double VO, VA, FREQ, TD, THETA;
                        double PHASE;
                        double phase;

                        PHASE = here->ISRCfunctionOrder > 5
                           ? here->ISRCcoeffs[5] : 0.0;

                        /* compute phase in radians */
                        phase = PHASE * M_PI / 180.0;

                        VO = here->ISRCcoeffs[0];
                        VA = here->ISRCcoeffs[1];
                        FREQ =  here->ISRCfunctionOrder > 2
                           && here->ISRCcoeffs[2] != 0.0
                           ? here->ISRCcoeffs[2] : (1/ckt->CKTfinalTime);
                        TD = here->ISRCfunctionOrder > 3
                           ? here->ISRCcoeffs[3] : 0.0;
                        THETA = here->ISRCfunctionOrder > 4
                           ? here->ISRCcoeffs[4] : 0.0;

                        time -= TD;
                        if (time <= 0) {

                            value = VO + VA * sin(phase);
                        } else {
                            value = VO + VA * sin(FREQ*time * 2.0 * M_PI + phase) *
                                exp(-time*THETA);
                        }
                    }
                    break;

                    case EXP: {
                        double V1, V2, TD1, TD2, TAU1, TAU2;

                        V1  = here->ISRCcoeffs[0];
                        V2  = here->ISRCcoeffs[1];
                        TD1 = here->ISRCfunctionOrder > 2
                           && here->ISRCcoeffs[2] != 0.0
                           ? here->ISRCcoeffs[2] : ckt->CKTstep;
                        TAU1 = here->ISRCfunctionOrder > 3
                           && here->ISRCcoeffs[3] != 0.0
                           ? here->ISRCcoeffs[3] : ckt->CKTstep;
                        TD2  = here->ISRCfunctionOrder > 4
                           && here->ISRCcoeffs[4] != 0.0
                           ? here->ISRCcoeffs[4] : TD1 + ckt->CKTstep;
                        TAU2 = here->ISRCfunctionOrder > 5
                           && here->ISRCcoeffs[5]
                           ? here->ISRCcoeffs[5] : ckt->CKTstep;

                        if(time <= TD1)  {
                            value = V1;
                        } else if (time <= TD2) {
                            value = V1 + (V2-V1)*(1-exp(-(time-TD1)/TAU1));
                        } else {
                            value = V1 + (V2-V1)*(1-exp(-(time-TD1)/TAU1)) +
                                         (V1-V2)*(1-exp(-(time-TD2)/TAU2)) ;
                        }
                    }
                    break;

                    case SFFM: {

                        double VO, VA, FC, MDI, FS;
                        double PHASEC, PHASES;
                        double phasec;
                        double phases;

                        PHASEC = here->ISRCfunctionOrder > 5
                            ? here->ISRCcoeffs[5] : 0.0;
                        PHASES = here->ISRCfunctionOrder > 6
                            ? here->ISRCcoeffs[6] : 0.0;

                        /* compute phases in radians */
                        phasec = PHASEC * M_PI / 180.0;
                        phases = PHASES * M_PI / 180.0;

                        VO = here->ISRCcoeffs[0];
                        VA = here->ISRCcoeffs[1];
                        FC = here->ISRCfunctionOrder > 2
                           && here->ISRCcoeffs[2]
                           ? here->ISRCcoeffs[2] : (1/ckt->CKTfinalTime);
                        MDI = here->ISRCfunctionOrder > 3
                           ? here->ISRCcoeffs[3] : 0.0;
                        FS  = here->ISRCfunctionOrder > 4
                           && here->ISRCcoeffs[4]
                           ? here->ISRCcoeffs[4] : (1/ckt->CKTfinalTime);

                        /* compute waveform value */
                        value = VO + VA *
                            sin((2.0 * M_PI * FC * time + phasec) +
                            MDI * sin(2.0 * M_PI * FS * time + phases));
                    }
                    break;

                    case AM: {

                        double VA, FC, MF, VO, TD;
                        double PHASEC, PHASES;
                        double phasec;
                        double phases;

                        PHASEC = here->ISRCfunctionOrder > 5
                            ? here->ISRCcoeffs[5] : 0.0;
                        PHASES = here->ISRCfunctionOrder > 6
                            ? here->ISRCcoeffs[6] : 0.0;

                        /* compute phases in radians */
                        phasec = PHASEC * M_PI / 180.0;
                        phases = PHASES * M_PI / 180.0;

                        VA = here->ISRCcoeffs[0];
                        VO = here->ISRCcoeffs[1];
                        MF = here->ISRCfunctionOrder > 2
                           && here->ISRCcoeffs[2]
                           ? here->ISRCcoeffs[2] : (1/ckt->CKTfinalTime);
                        FC = here->ISRCfunctionOrder > 3
                           ? here->ISRCcoeffs[3] : 0.0;
                        TD  = here->ISRCfunctionOrder > 4
                           && here->ISRCcoeffs[4]
                           ? here->ISRCcoeffs[4] : 0.0;

                        time -= TD;
                        if (time <= 0) {
                            value = 0;
                        } else {
                            /* compute waveform value */
                            value = VA * (VO + sin(2.0 * M_PI * MF * time + phases )) *
                                sin(2.0 * M_PI * FC * time + phases);
                        }
                    }
                    break;

                    case PWL: {
                        int i;
                        if(time < *(here->ISRCcoeffs)) {
                            value = *(here->ISRCcoeffs + 1) ;
                            break;
                        }
                        for(i=0; i < (here->ISRCfunctionOrder / 2) - 1; i++) {
                            if(*(here->ISRCcoeffs+2*i)==time) {
                                value = *(here->ISRCcoeffs+2*i+1);
                                goto loadDone;
                            }
                            if((*(here->ISRCcoeffs+2*i)<time) &&
                               (*(here->ISRCcoeffs+2*(i+1)) >time)) {
                                value = *(here->ISRCcoeffs+2*i+1) +
                                   (((time-*(here->ISRCcoeffs+2*i))/
                                   (*(here->ISRCcoeffs+2*(i+1)) -
                                    *(here->ISRCcoeffs+2*i))) *
                                   (*(here->ISRCcoeffs+2*i+3) -
                                    *(here->ISRCcoeffs+2*i+1)));
                                goto loadDone;
                            }
                        }
                        value = *(here->ISRCcoeffs+ here->ISRCfunctionOrder-1) ;
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
                        double RTSAM = state->RTSAM;

                        /* reset top (hack for repeated tran commands)
                        when there is the jump from time=0 to time>0 */
                        if (time == 0.0)
                            state->timezero = TRUE;
                        else
                            if (state->timezero) {
                                state->top = 0;
                                state->timezero = FALSE;
                            }

                        /* no noise or time == 0 */
                        if (TS == 0.0 || time == 0.0) {
                            value = 0.0;
                        }
                        else {

                            /* 1/f and white noise */
                            size_t n1 = (size_t)floor(time / TS);

                            double V1 = trnoise_state_get(state, ckt, n1);
                            double V2 = trnoise_state_get(state, ckt, n1 + 1);

                            value = V1 + (V2 - V1) * (time / TS - (double)n1);
                        }

                        /* RTS noise */
                        if (RTSAM > 0) {
                            double RTScapTime = state->RTScapTime;
                            if (time >= RTScapTime)
                                value += RTSAM;
                        }

                        /* DC value */
                        if(here -> ISRCdcGiven)
                            value += here->ISRCdcValue;
                    }
                    break;

                    case TRRANDOM: {
                        struct trrandom_state *state = here -> ISRCtrrandom_state;
                        value = state -> value;
                        /* DC value */
                        if(here -> ISRCdcGiven)
                            value += here->ISRCdcValue;
                    }
                    break;

#ifdef SHARED_MODULE
                    case EXTERNAL: {
                        value = getisrcval(time, here->ISRCname);
                        if(here -> ISRCdcGiven)
                            value += here->ISRCdcValue;
                    }
                    break;
#endif

                } // switch
            } // else (line 48)
loadDone:

/* gtri - begin - wbk - modify for supply ramping option */
#ifdef XSPICE_EXP
            value *= ckt->CKTsrcFact;
            value *= cm_analog_ramp_factor();
#else
            if (ckt->CKTmode & MODETRANOP)
                value *= ckt->CKTsrcFact;
#endif
/* gtri - end - wbk - modify for supply ramping option */

            *(ckt->CKTrhs + (here->ISRCposNode)) += m * value;
            *(ckt->CKTrhs + (here->ISRCnegNode)) -= m * value;

/* gtri - end - wbk - modify to process srcFact, etc. for all sources */

            here->ISRCcurrent = m * value;
        } // for loop instances
    } // for loop models

    return(OK);
}

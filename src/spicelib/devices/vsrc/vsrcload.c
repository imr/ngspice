/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vsrcdefs.h"
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
extern double getvsrcval(double, char*);
#endif

int
VSRCload(GENmodel *inModel, CKTcircuit *ckt)
        /* actually load the current value into the
         * sparse matrix previously provided
         */
{
    VSRCmodel *model = (VSRCmodel *) inModel;
    VSRCinstance *here;
    double time;
    double value = 0.0;

    /*  loop through all the source models */
    for( ; model != NULL; model = VSRCnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = VSRCinstances(model); here != NULL ;
                here=VSRCnextInstance(here)) {

#ifndef RFSPICE
            * (here->VSRCposIbrPtr) += 1.0;
            *(here->VSRCnegIbrPtr) -= 1.0;
            *(here->VSRCibrPosPtr) += 1.0;
            *(here->VSRCibrNegPtr) -= 1.0;
#else
            if (here->VSRCisPort)
            {
                // here->VSRCcurrent = (*(ckt->CKTrhs[Old] + (here->VSRCbranch))

                *(here->VSRCposIbrPtr) += 1.0;
                *(here->VSRCnegIbrPtr) -= 1.0;
                *(here->VSRCibrPosPtr) += 1.0;
                *(here->VSRCibrNegPtr) -= 1.0;

                double g0 = here->VSRCportY0;
                *(here->VSRCposPosPtr) += g0;
                *(here->VSRCnegNegPtr) += g0;
                *(here->VSRCposNegPtr) -= g0;
                *(here->VSRCnegPosPtr) -= g0;
            }
            else
            {
                *(here->VSRCposIbrPtr) += 1.0;
                *(here->VSRCnegIbrPtr) -= 1.0;
                *(here->VSRCibrPosPtr) += 1.0;
                *(here->VSRCibrNegPtr) -= 1.0;
            }
#endif

            if( (ckt->CKTmode & (MODEDCOP | MODEDCTRANCURVE)) &&
                    here->VSRCdcGiven ) {
                /* load using DC value */
#ifdef XSPICE_EXP
/* gtri - begin - wbk - modify to process srcFact, etc. for all sources */
                value = here->VSRCdcValue;
#else
                value = here->VSRCdcValue * ckt->CKTsrcFact;
#endif
            } else {
                if(ckt->CKTmode & (MODEDC)) {
                    time = 0;
                } else {
                    time = ckt->CKTtime;
                }
                /* use the transient functions */
                switch(here->VSRCfunctionType) {

                    default:
                        value = here->VSRCdcValue;
                        break;

                    case PULSE: {
                        double V1, V2, TD, TR, TF, PW, PER;
                        double basetime = 0;
                        double PHASE;
                        double phase;
                        double deltat;
                        double tmax = 1e99;

                        V1 = here->VSRCcoeffs[0];
                        V2 = here->VSRCcoeffs[1];
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

                        /* shift time by delay time TD */
                        time -=  TD;

                        PHASE = here->VSRCfunctionOrder > 7
                           ? here->VSRCcoeffs[7] : 0.0;

                        if (newcompat.xs) { /* 7th parameter is PHASE */
                            /* normalize phase to cycles */
                            phase = PHASE / 360.0;
                            phase = fmod(phase, 1.0);
                            deltat = phase * PER;
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

                        PHASE = here->VSRCfunctionOrder > 5
                           ? here->VSRCcoeffs[5] : 0.0;

                        /* compute phase in radians */
                        phase = PHASE * M_PI / 180.0;

                        VO = here->VSRCcoeffs[0];
                        VA = here->VSRCcoeffs[1];
                        FREQ =  here->VSRCfunctionOrder > 2
                           && here->VSRCcoeffs[2] != 0.0
                           ? here->VSRCcoeffs[2] : (1/ckt->CKTfinalTime);
                        TD = here->VSRCfunctionOrder > 3
                           ? here->VSRCcoeffs[3] : 0.0;
                        THETA = here->VSRCfunctionOrder > 4
                           ? here->VSRCcoeffs[4] : 0.0;

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

                        V1  = here->VSRCcoeffs[0];
                        V2  = here->VSRCcoeffs[1];
                        TD1 = here->VSRCfunctionOrder > 2
                           && here->VSRCcoeffs[2] != 0.0
                           ? here->VSRCcoeffs[2] : ckt->CKTstep;
                        TAU1 = here->VSRCfunctionOrder > 3
                           && here->VSRCcoeffs[3] != 0.0
                           ? here->VSRCcoeffs[3] : ckt->CKTstep;
                        TD2  = here->VSRCfunctionOrder > 4
                           && here->VSRCcoeffs[4] != 0.0
                           ? here->VSRCcoeffs[4] : TD1 + ckt->CKTstep;
                        TAU2 = here->VSRCfunctionOrder > 5
                           && here->VSRCcoeffs[5]
                           ? here->VSRCcoeffs[5] : ckt->CKTstep;

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

                        PHASEC = here->VSRCfunctionOrder > 5
                            ? here->VSRCcoeffs[5] : 0.0;
                        PHASES = here->VSRCfunctionOrder > 6
                            ? here->VSRCcoeffs[6] : 0.0;

                        /* compute phases in radians */
                        phasec = PHASEC * M_PI / 180.0;
                        phases = PHASES * M_PI / 180.0;

                        VO = here->VSRCcoeffs[0];
                        VA = here->VSRCcoeffs[1];
                        FC = here->VSRCfunctionOrder > 2
                           && here->VSRCcoeffs[2]
                           ? here->VSRCcoeffs[2] : (1/ckt->CKTfinalTime);
                        MDI = here->VSRCfunctionOrder > 3
                           ? here->VSRCcoeffs[3] : 0.0;
                        FS  = here->VSRCfunctionOrder > 4
                           && here->VSRCcoeffs[4]
                           ? here->VSRCcoeffs[4] : (1/ckt->CKTfinalTime);

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

                        PHASEC = here->VSRCfunctionOrder > 5
                            ? here->VSRCcoeffs[5] : 0.0;
                        PHASES = here->VSRCfunctionOrder > 6
                            ? here->VSRCcoeffs[6] : 0.0;

                        /* compute phases in radians */
                        phasec = PHASEC * M_PI / 180.0;
                        phases = PHASES * M_PI / 180.0;

                        VA = here->VSRCcoeffs[0];
                        VO = here->VSRCcoeffs[1];
                        MF = here->VSRCfunctionOrder > 2
                           && here->VSRCcoeffs[2]
                           ? here->VSRCcoeffs[2] : (1/ckt->CKTfinalTime);
                        FC = here->VSRCfunctionOrder > 3
                           ? here->VSRCcoeffs[3] : 0.0;
                        TD  = here->VSRCfunctionOrder > 4
                           && here->VSRCcoeffs[4]
                           ? here->VSRCcoeffs[4] : 0.0;

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
                        int    i;
                        double end_time, itime;

                        time -= here->VSRCrdelay;
                        if (time < here->VSRCcoeffs[0]) {
                            value = here->VSRCcoeffs[1];
                            value = value;
                            break;
                        }

                        end_time =
                            here->VSRCcoeffs[here->VSRCfunctionOrder - 2];
                        if (time > end_time) {
                            double period;

                            if (here->VSRCrGiven) {
                                /* Repeating. */

                                period = end_time -
                                    here->VSRCcoeffs[here->VSRCrBreakpt];
                                time -= here->VSRCcoeffs[here->VSRCrBreakpt];
                                time -= period * floor(time / period);
                                time += here->VSRCcoeffs[here->VSRCrBreakpt];
                            } else {
                                value =
                                    here->VSRCcoeffs[here->VSRCfunctionOrder - 1];
                                break;
                            }
                        }

                        for (i = 2;  i < here->VSRCfunctionOrder; i += 2) {
                            itime = here->VSRCcoeffs[i];
                            if (itime >= time) {
                                time -= here->VSRCcoeffs[i - 2];
                                time /=  here->VSRCcoeffs[i] -
                                             here->VSRCcoeffs[i - 2];
                                value = here->VSRCcoeffs[i - 1];
                                value += time *
                                    ( here->VSRCcoeffs[i + 1] -
                                      here->VSRCcoeffs[i - 1]);
                                break;
                            }
                        }
                        break;
                    }

/**** tansient noise routines:
VNoi2 2 0  DC 0 TRNOISE(10n 0.5n 0 0n) : generate gaussian distributed noise
                        rms value, time step, 0 0
VNoi1 1 0  DC 0 TRNOISE(0n 0.5n 1 10n) : generate 1/f noise
                        0,  time step, exponent < 2, rms value

VNoi3 3 0  DC 0 TRNOISE(0 0 0 0 15m 22u 50u) : generate RTS noise
                        0 0 0 0, amplitude, capture time, emission time
*/
                    case TRNOISE: {

                        struct trnoise_state *state = here -> VSRCtrnoise_state;

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
                        if(TS == 0.0 || time == 0.0) {
                            value = 0.0;
                        } else {

                            /* 1/f and white noise */
                            size_t n1 = (size_t) floor(time / TS);

                            double V1 = trnoise_state_get(state, ckt, n1);
                            double V2 = trnoise_state_get(state, ckt, n1+1);

                            value = V1 + (V2 - V1) * (time / TS - (double)n1);
                        }

                        /* RTS noise */
                        if (RTSAM > 0) {
                            double RTScapTime = state->RTScapTime;
                            if (time >= RTScapTime)
                                value += RTSAM;
                        }

                        /* DC value */
                        if(here -> VSRCdcGiven)
                            value += here->VSRCdcValue;
                    }
                    break;

                    case TRRANDOM: {
                        struct trrandom_state *state = here -> VSRCtrrandom_state;
                        value = state -> value;
                        /* DC value */
                        if(here -> VSRCdcGiven)
                            value += here->VSRCdcValue;
                    }
                    break;

#ifdef SHARED_MODULE
                    case EXTERNAL: {
                        value = getvsrcval(time, here->VSRCname);
                        if(here -> VSRCdcGiven)
                            value += here->VSRCdcValue;
                    }
                    break;
#endif
#ifdef RFSPICE
                    case PORT:
                    {
                        value += here->VSRCVAmplitude * cos(time * here->VSRC2pifreq);

                    }
#endif

                } // switch
            } // else (line 48)

/* gtri - begin - wbk - modify for supply ramping option */
#ifdef XSPICE_EXP
            value *= ckt->CKTsrcFact;
            value *= cm_analog_ramp_factor();
#else
            if (ckt->CKTmode & MODETRANOP)
                value *= ckt->CKTsrcFact;
#endif
/* gtri - end - wbk - modify to process srcFact, etc. for all sources */

            /* load the new voltage value into the matrix */
            *(ckt->CKTrhs + (here->VSRCbranch)) += value;

        } // for loop instances
    } // for loop models

    return(OK);
}

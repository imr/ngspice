/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "isrcdefs.h"
#include "trandefs.h"
#include "sperror.h"
#include "suffix.h"
#include "missing_math.h"
#include "1-f-code.h"

extern int fftInit(long M);
extern void fftFree(void);
extern void rffts(float *data, long M, long Rows);

int
ISRCaccept(CKTcircuit *ckt, GENmodel *inModel)
        /* set up the breakpoint table.  */
{
    ISRCmodel *model = (ISRCmodel*)inModel;
    ISRCinstance *here;
    int error;

    /*  loop through all the voltage source models */
    for( ; model != NULL; model = model->ISRCnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->ISRCinstances; here != NULL ;
                here=here->ISRCnextInstance) {
            
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

#define SAMETIME(a,b) (fabs((a)-(b))<= TIMETOL * PW)
#define TIMETOL 1e-7

                        double TD, TR, TF, PW, PER;

/* gtri - begin - wbk - add PHASE parameter */
#ifdef XSPICE		    
                        double  PHASE;
                        double phase;
                        double deltat;
                        double basephase;
#endif		    
                        double time;
                        double basetime = 0;

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
#ifdef XSPICE
                        PHASE = here->ISRCfunctionOrder > 8 
                           ? here->ISRCcoeffs[7] : 0.0;
#endif
                        /* offset time by delay  and limit to zero */
                        time = ckt->CKTtime - TD;

#ifdef XSPICE
                        if(time < 0.0)
                           time = 0.0;
#endif

#ifdef XSPICE			
                        /* normalize phase to 0 - 2PI */ 
                        phase = PHASE * M_PI / 180.0;
                        basephase = 2 * M_PI * floor(phase / (2 * M_PI));
                        phase -= basephase;

                        /* compute equivalent delta time and add to time */
                        deltat = (phase / (2 * M_PI)) * PER;
                        time += deltat;	
#endif		    
/* gtri - end - wbk - add PHASE parameter */
                
                        if(time >= PER) {
                            /* repeating signal - figure out where we are */
                            /* in period */
                            basetime = PER * floor(time/PER);
                            time -= basetime;
                        }

                        if( time <= 0.0 || time >= TR + PW + TF) {
                            if(ckt->CKTbreak &&  SAMETIME(time,0.0)) {
                                /* set next breakpoint */
                                error = CKTsetBreak(ckt,basetime + TR + TD);
                                if(error) return(error);
                            } else if(ckt->CKTbreak && SAMETIME(TR+PW+TF,time) ) {
                                /* set next breakpoint */
                                error = CKTsetBreak(ckt,basetime + PER + TD);
                                if(error) return(error);
                            } else if (ckt->CKTbreak && (time == -TD) ) {
                                /* set next breakpoint */
                                error = CKTsetBreak(ckt,basetime + TD);
                                if(error) return(error);
                            } else if (ckt->CKTbreak && SAMETIME(PER,time) ) {
                                /* set next breakpoint */
                                error = CKTsetBreak(ckt,basetime + TD + TR + PER);
                                if(error) return(error);
                            }
                        } else  if ( time >= TR && time <= TR + PW) {
                            if(ckt->CKTbreak &&  SAMETIME(time,TR) ) {
                                /* set next breakpoint */
                                error = CKTsetBreak(ckt,basetime + TD+TR + PW);
                                if(error) return(error);
                            } else if(ckt->CKTbreak &&  SAMETIME(TR+PW,time) ) {
                                /* set next breakpoint */
                                error = CKTsetBreak(ckt,basetime + TD+TR + PW + TF);
                                if(error) return(error);
                            }
                        } else if (time > 0 && time < TR) {
                            if(ckt->CKTbreak && SAMETIME(time,0) ) {
                                /* set next breakpoint */
                                error = CKTsetBreak(ckt,basetime + TD+TR);
                                if(error) return(error);
                            } else if(ckt->CKTbreak && SAMETIME(time,TR)) {
                                /* set next breakpoint */
                                error = CKTsetBreak(ckt,basetime + TD+TR + PW);
                                if(error) return(error);
                            }
                        } else { /* time > TR + PW && < TR + PW + TF */
                            if(ckt->CKTbreak && SAMETIME(time,TR+PW) ) {
                                /* set next breakpoint */
                                error = CKTsetBreak(ckt,basetime + TD+TR + PW +TF);
                                if(error) return(error);
                            } else if(ckt->CKTbreak && SAMETIME(time,TR+PW+TF) ) {
                                /* set next breakpoint */
                                error = CKTsetBreak(ckt,basetime + TD+PER);
                                if(error) return(error);
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
                            if((*(here->ISRCcoeffs+2*i)==ckt->CKTtime)) {
                                if(ckt->CKTbreak) {
                                    error = CKTsetBreak(ckt,
                                            *(here->ISRCcoeffs+2*i+2));
                                    if(error) return(error);
                                }
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

                        if (TS == 0.0) // no further breakpoint if value not given
                            break;

                        /* FIXME, dont' want this here, over to aof_get or somesuch */
                        if (ckt->CKTtime == 0.0) {
                            printf("VSRC: free fft tables\n");
                            fftFree();
                        }

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
                    }
                    break;

                }
            }
bkptset: ;
        }
    }
    return(OK);
}

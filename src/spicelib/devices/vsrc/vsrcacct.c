/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "vsrcdefs.h"
#include "trandefs.h"
#include "sperror.h"
#include "suffix.h"
#include "missing_math.h"

#define SAMETIME(a,b)    (fabs((a)-(b))<= TIMETOL * PW)
#define TIMETOL    1e-7
		
int
VSRCaccept(CKTcircuit *ckt, GENmodel *inModel)
        /* set up the breakpoint table.
         */
{
    VSRCmodel *model = (VSRCmodel *)inModel;
    VSRCinstance *here;
    int error;

    /*  loop through all the voltage source models */
    for( ; model != NULL; model = model->VSRCnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->VSRCinstances; here != NULL ;
                here=here->VSRCnextInstance) {
            
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
                    double tshift;
                    double time = 0.;
                    double basetime = 0;

/* gtri - begin - wbk - add PHASE parameter */
#ifdef XSPICE		    
                    double PHASE;
                    double phase;
                    double deltat;
#endif		    		    
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
#ifdef XSPICE
                    PHASE = here->VSRCfunctionOrder > 7 
                        ? here->VSRCcoeffs[7] : 0.0;
#endif
                    /* offset time by delay */
                    time = ckt->CKTtime - TD;
                    tshift = TD;
#ifdef XSPICE					
		         /* normalize phase to 0 - 360° */ 
		         /* normalize phase to cycles */
                    phase = PHASE / 360.0;
                    phase = fmod(phase, 1.0);
                    deltat =  phase * PER;
                    while (deltat > 0) 
                        deltat -= PER;
                    time += deltat;
                    tshift = TD - deltat; 
#endif		    
/* gtri - end - wbk - add PHASE parameter */			
		   		   
                    if(time >= PER) {
                        /* repeating signal - figure out where we are */
                        /* in period */
                        basetime = PER * floor(time/PER);
                        time -= basetime;
                    }
                    if( time <= 0 || time >= TR + PW + TF) {
                        if(ckt->CKTbreak &&  SAMETIME(time,0)) {
                            /* set next breakpoint */
                            error = CKTsetBreak(ckt,basetime + TR + tshift);
                            if(error) return(error);
                        } else if(ckt->CKTbreak && SAMETIME(TR+PW+TF,time) ) {
                            /* set next breakpoint */
                            error = CKTsetBreak(ckt,basetime + PER + tshift);
                            if(error) return(error);
                        } else if (ckt->CKTbreak && (time == -tshift) ) {
                            /* set next breakpoint */
                            error = CKTsetBreak(ckt,basetime + tshift);
                            if(error) return(error);
                        } else if (ckt->CKTbreak && SAMETIME(PER,time) ) {
                            /* set next breakpoint */
                            error = CKTsetBreak(ckt,basetime + tshift + TR + PER);
                            if(error) return(error);
                        }
                    } else  if ( time >= TR && time <= TR + PW) {
                        if(ckt->CKTbreak &&  SAMETIME(time,TR) ) {
                            /* set next breakpoint */
                            error = CKTsetBreak(ckt,basetime + tshift + TR + PW);
                            if(error) return(error);
                        } else if(ckt->CKTbreak &&  SAMETIME(TR+PW,time) ) {
                            /* set next breakpoint */
                            error = CKTsetBreak(ckt,basetime + tshift + TR + PW + TF);
                            if(error) return(error);
                        }
                    } else if (time > 0 && time < TR) {
                        if(ckt->CKTbreak && SAMETIME(time,0) ) {
                            /* set next breakpoint */
                            error = CKTsetBreak(ckt,basetime + tshift + TR);
                            if(error) return(error);
                        } else if(ckt->CKTbreak && SAMETIME(time,TR)) {
                            /* set next breakpoint */
                            error = CKTsetBreak(ckt,basetime + tshift + TR + PW);
                            if(error) return(error);
                        }
                    } else { /* time > TR + PW && < TR + PW + TF */
                        if(ckt->CKTbreak && SAMETIME(time,TR+PW) ) {
                            /* set next breakpoint */
                            error = CKTsetBreak(ckt,basetime + tshift+TR + PW +TF);
                            if(error) return(error);
                        } else if(ckt->CKTbreak && SAMETIME(time,TR+PW+TF) ) {
                            /* set next breakpoint */
                            error = CKTsetBreak(ckt,basetime + tshift + PER);
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
                    if(ckt->CKTtime < *(here->VSRCcoeffs)) {
                        if(ckt->CKTbreak) {
                            error = CKTsetBreak(ckt,*(here->VSRCcoeffs));
                            break;
                        }
                    }
                    for(i=0;i<(here->VSRCfunctionOrder/2)-1;i++) {
/*		      if((*(here->VSRCcoeffs+2*i)==ckt->CKTtime)) {
		           if(ckt->CKTbreak) {*/
                        if ( ckt->CKTbreak && AlmostEqualUlps(*(here->VSRCcoeffs+2*i), ckt->CKTtime, 3 ) ) {
                            error = CKTsetBreak(ckt, *(here->VSRCcoeffs+2*i+2));
                            if(error) return(error);
                            goto bkptset;
                        } 
                    }
                    break;
                }
                }
            }
bkptset: ;
        }
    }
    return(OK);
}

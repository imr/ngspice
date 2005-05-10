/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 Alansfixes
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "isrcdefs.h"
#include "trandefs.h"
#include "sperror.h"
#include "suffix.h"

#ifdef XSPICE_EXP
#include "cmproto.h"
#endif

int
ISRCload(GENmodel *inModel, CKTcircuit *ckt)
        /* actually load the current current value into the 
         * sparse matrix previously provided 
         */
{
    ISRCmodel *model = (ISRCmodel*)inModel;
    ISRCinstance *here;
    double value;
    double time;

    /*  loop through all the current source models */
    for( ; model != NULL; model = model->ISRCnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->ISRCinstances; here != NULL ;
                here=here->ISRCnextInstance) {
	    if (here->ISRCowner != ARCHme) continue;
            
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
                /* use transient function */
                switch(here->ISRCfunctionType) {

                case PULSE: {
		    double	V1, V2, TD, TR, TF, PW, PER;		    
                    double	basetime = 0;
/* gtri - begin - wbk - add PHASE parameter */
#ifdef XSPICE
                    double PHASE;
                    double phase;
                    double deltat;
                    double basephase;
		    
		    PHASE = here->ISRCfunctionOrder > 7
		         ? here->ISRCcoeffs[7] : 0.0;
			 
		    /* normalize phase to 0 - 2PI */ 
                    phase = PHASE * M_PI / 180.0;
                    basephase = 2 * M_PI * floor(phase / (2 * M_PI));
                    phase -= basephase;

                    /* compute equivalent delta time and add to time */
                    deltat = (phase / (2 * M_PI)) * PER;
                    time += deltat; 
#endif
/* gtri - end - wbk - add PHASE parameter */		    

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

                    time -= TD;

                    if(time > PER) {
                        /* repeating signal - figure out where we are */
                        /* in period */
                        basetime = PER * floor(time/PER);
                        time -= basetime;
                    }
                    if( time <= 0 || time >= TR + PW + TF) {
                        value = V1;
                    } else  if ( time >= TR && time <= TR + PW) {
                        value = V2;
                    } else if (time > 0 && time < TR) {
                        value = V1 + (V2 - V1) * (time) / TR;
                    } else { /* time > TR + PW && < TR + PW + TF */
                        value = V2 + (V1 - V2) * (time - (TR + PW)) / TF;
                    }
                }
                break;

                case SINE: {
		
		    double VO, VA, FREQ, TD, THETA;
/* gtri - begin - wbk - add PHASE parameter */
#ifdef XSPICE
                    double PHASE;
		    double phase;

                    PHASE = here->ISRCfunctionOrder > 5
		           ? here->ISRCcoeffs[5] : 0.0;
		       		
	     	    /* compute phase in radians */ 
                    phase = PHASE * M_PI / 180.0;
#endif
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
                        value = VO;
                    } else {
#ifdef XSPICE
                        value = VO + VA * sin(FREQ*time * 2.0 * M_PI + phase) * 
                                exp(-time*THETA);
#else		    
                        value = VO + VA * sin(FREQ*time * 2.0 * M_PI) * 
                                exp(-time*THETA);
#endif
/* gtri - end - wbk - add PHASE parameter */			
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
                case SFFM:{
		
		    double VO, VA, FC, MDI, FS;
/* gtri - begin - wbk - add PHASE parameters */
#ifdef XSPICE

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

#endif		    
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

#ifdef XSPICE
                    /* compute waveform value */
                    value = VO + VA * 
                        sin((2.0 * M_PI * FC * time + phasec) +
                        MDI * sin(2.0 * M_PI * FS * time + phases));
#else /* XSPICE */
                    value = VO + VA * 
                        sin((2.0 * M_PI * FC * time) +
                        MDI * sin(2.0 * M_PI * FS * time));
#endif /* XSPICE */
/* gtri - end - wbk - add PHASE parameters */

                }
                break;
		case AM:{
		
		double VA, FC, MF, VO, TD;
/* gtri - begin - wbk - add PHASE parameters */
#ifdef XSPICE

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

#endif			
		
		   VA = here->ISRCcoeffs[0];
                   VO = here->ISRCcoeffs[1];
                   MF = here->ISRCfunctionOrder > 2 
		      && here->ISRCcoeffs[2]
		       ? here->ISRCcoeffs[2] : (1/ckt->CKTfinalTime);
                   FC = here->ISRCfunctionOrder > 3
		        ? here->ISRCcoeffs[3] : 0.0;
                   TD  = here->ISRCfunctionOrder > 4 
		       && here->ISRCcoeffs[4]
		        ? here->ISRCcoeffs[4] : 0,0;

                    time -= TD;
                    if (time <= 0) {
                        value = 0;
                    } else {
#ifdef XSPICE
                    /* compute waveform value */
		    value = VA * (VO + sin(2.0 * M_PI * MF * time + phases )) *
		        sin(2 * M_PI * FC * time + phases);
                    
#else /* XSPICE */		    
		        value = VA * (VO + sin(2.0 * M_PI * MF * time)) *
		        sin(2 * M_PI * FC * time);
#endif			
		    }
		    
/* gtri - end - wbk - add PHASE parameters */
		}
		break;
                default:
#ifdef XSPICE_EXP
                    value = here->ISRCdcValue;
#else
                    value = here->ISRCdcValue * ckt->CKTsrcFact;
#endif
                    break;
                case PWL: {
                    int i;
                    if(time< *(here->ISRCcoeffs)) {
                        value = *(here->ISRCcoeffs + 1) ;
                        break;
                    }
                    for(i=0;i<=(here->ISRCfunctionOrder/2)-1;i++) {
                        if((*(here->ISRCcoeffs+2*i)==time)) {
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
                }
            }
loadDone:

/* gtri - begin - wbk - modify for supply ramping option */
#ifdef XSPICE_EXP
            value *= ckt->CKTsrcFact;
            value *= cm_analog_ramp_factor();

#else
            if (ckt->CKTmode & MODETRANOP) value *= ckt->CKTsrcFact;
#endif
/* gtri - end - wbk - modify for supply ramping option */

            *(ckt->CKTrhs + (here->ISRCposNode)) += value;
            *(ckt->CKTrhs + (here->ISRCnegNode)) -= value;

/* gtri - end - wbk - modify to process srcFact, etc. for all sources */

#ifdef XSPICE
/* gtri - begin - wbk - record value so it can be output if requested */
            here->ISRCcurrent = value;
/* gtri - end   - wbk - record value so it can be output if requested */
#endif

        }
    }
    return(OK);
}

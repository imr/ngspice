/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 Alansfixes
**********/

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "isrcdefs.h"
#include "trandefs.h"
#include "sperror.h"
#include "suffix.h"

int
ISRCload(inModel,ckt)
    GENmodel *inModel;
    CKTcircuit *ckt;
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
                value = here->ISRCdcValue * ckt->CKTsrcFact;
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
#define VO    (*(here->ISRCcoeffs))
#define VA    (*(here->ISRCcoeffs+1))
#define FREQ  (((here->ISRCfunctionOrder >=3) && (*(here->ISRCcoeffs+2)))? \
    (*(here->ISRCcoeffs+2)):(1/ckt->CKTfinalTime))
#define TD    ((here->ISRCfunctionOrder >=4)?(*(here->ISRCcoeffs+3)):(0.0))
#define THETA ((here->ISRCfunctionOrder >=5)?(*(here->ISRCcoeffs+4)):(0.0))
                    time -= TD;
                    if (time <= 0) {
                        value = VO;
                    } else {
                        value = VO + VA * sin(FREQ*time * 2.0 * M_PI) * 
                        exp(-time*THETA);
                    }
#undef VO
#undef VA
#undef FREQ
#undef TD
#undef THETA
                }
                break;
                case EXP: {
                    double td1;
                    double td2;
#define V1 (*(here->ISRCcoeffs))
#define V2 (*(here->ISRCcoeffs+1))
#define TD1 ((here->ISRCfunctionOrder >=3)?(*(here->ISRCcoeffs+2)):\
    ckt->CKTstep)
#define TAU1 (((here->ISRCfunctionOrder >=4) && (*(here->ISRCcoeffs+3)))? \
    (*(here->ISRCcoeffs+3)):ckt->CKTstep)
#define TD2 (((here->ISRCfunctionOrder >=5) && (*(here->ISRCcoeffs+4)))? \
    (*(here->ISRCcoeffs+4)):TD1+ckt->CKTstep)
#define TAU2 (((here->ISRCfunctionOrder >=6) && (*(here->ISRCcoeffs+5)))? \
    (*(here->ISRCcoeffs+5)):ckt->CKTstep)
                    td1 = TD1;
                    td2 = TD2;
                    if(time <= td1)  {
                        value = V1;
                    } else if (time <= td2) {
                        value = V1 + (V2-V1)*(1-exp(-(time-td1)/TAU1));
                    } else {
                        value = V1 + (V2-V1)*(1-exp(-(time-td1)/TAU1)) +
                                     (V1-V2)*(1-exp(-(time-td2)/TAU2)) ;
                    }
#undef V1
#undef V2
#undef TD1
#undef TAU1
#undef TD2
#undef TAU2
                }
                break;
                case SFFM:{
#define VO (*(here->ISRCcoeffs))
#define VA (*(here->ISRCcoeffs+1))
#define FC (((here->ISRCfunctionOrder >=3) && (*(here->ISRCcoeffs+2)))? \
    (*(here->ISRCcoeffs+2)):(1/ckt->CKTfinalTime))
#define MDI ((here->ISRCfunctionOrder>=4)?(*(here->ISRCcoeffs+3)):0.0)
#define FS (((here->ISRCfunctionOrder >=5) && (*(here->ISRCcoeffs+4)))? \
    (*(here->ISRCcoeffs+4)):(1/ckt->CKTfinalTime))
                    value = VO + VA * 
                        sin((2.0 * M_PI * FC * time) +
                        MDI * sin(2.0 * M_PI * FS * time));
#undef VO
#undef VA
#undef FC
#undef MDI
#undef FS
                }
                break;
                default:
                    value = here->ISRCdcValue * ckt->CKTsrcFact;
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
            if (ckt->CKTmode & MODETRANOP) value *= ckt->CKTsrcFact;
            *(ckt->CKTrhs + (here->ISRCposNode)) += value;
            *(ckt->CKTrhs + (here->ISRCnegNode)) -= value;
        }
    }
    return(OK);
}

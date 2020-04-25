/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Jaijeet S Roychowdhury
**********/

/*
 * CKTdisto (ckt, mode)
 */


#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "vsrc/vsrcdefs.h"
#include "isrc/isrcdefs.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/distodef.h"
#include "ngspice/sperror.h"


int
CKTdisto (CKTcircuit *ckt, int mode)
{
    DISTOAN *job = (DISTOAN *) ckt->CKTcurJob;

    int i;
    int error=0;
    int size;

    /* FALLTHROUGH added to suppress GCC warning due to
     * -Wimplicit-fallthrough flag */
    switch(mode) {

    case D_SETUP:

	    for (i=0;i<DEVmaxnum;i++) {
		if ( DEVices[i] && DEVices[i]->DEVdisto && ckt->CKThead[i] ) {
		    error = DEVices[i]->DEVdisto (mode, ckt->CKThead[i], ckt);
		    if(error) return(error);
		}
	    }
	    break;

    case D_TWOF1:
    case D_THRF1:
    case D_F1PF2:
    case D_F1MF2:
    case D_2F1MF2:

	    size = SMPmatSize(ckt->CKTmatrix);
	    for (i=1; i<=size; i++)
	    {
	    ckt->CKTrhs[i] = 0.0;
	    ckt->CKTirhs[i] = 0.0;
	    }

	    for (i=0;i<DEVmaxnum;i++) {
		if ( DEVices[i] && DEVices[i]->DEVdisto && ckt->CKThead[i] ) {
		    error = DEVices[i]->DEVdisto (mode, ckt->CKThead[i], ckt);
		    if(error) return(error);
		}
	    }
	    break;

    case D_RHSF1:

 	   job->Df2given = 0; /* will change if any F2 source is found */
        /* FALLTHROUGH */
    case D_RHSF2:



{
    int vcode;
    int icode;
    double mag=0.0;
    double phase=0.0;

    size = SMPmatSize(ckt->CKTmatrix);
    for (i=0;i<=size;i++) {
        ckt->CKTrhs[i]=0;
        ckt->CKTirhs[i]=0;
	}

    vcode = CKTtypelook("Vsource");
    icode = CKTtypelook("Isource");


        if(vcode >= 0) {
            /* voltage sources are in this version, so use them */
            VSRCinstance *here;
            VSRCmodel *model;
            for(model = (VSRCmodel *)ckt->CKThead[vcode];model != NULL;
                    model=VSRCnextModel(model)){
                for(here=VSRCinstances(model);here!=NULL;
                        here=VSRCnextInstance(here)) {

/* check if the source has a distortion input*/

if (here->VSRCdGiven) {
  if (here->VSRCdF2given) job->Df2given = 1;
  if ((here->VSRCdF1given) && (mode == D_RHSF1)) {
 
     mag = here->VSRCdF1mag;
     phase = here->VSRCdF1phase;
}
else if ((here->VSRCdF2given) && (mode == D_RHSF2)) {

     mag = here->VSRCdF2mag;
     phase = here->VSRCdF2phase;
}
if (((here->VSRCdF1given) && (mode == D_RHSF1)) || 
	((here->VSRCdF2given) && (mode == D_RHSF2))) {
   
	ckt->CKTrhs[here->VSRCbranch] = 0.5*mag* cos(M_PI*phase/180.0);
	ckt->CKTirhs[here->VSRCbranch] = 0.5*mag*sin(M_PI*phase/180.0); 
}


                    }
                }
            }
        }
        if(icode >= 0 ) {
            /* current sources are in this version, so use them */
            ISRCinstance *here;
            ISRCmodel *model;

            for(model= (ISRCmodel *)ckt->CKThead[icode];model != NULL;
                    model=ISRCnextModel(model)){
                for(here=ISRCinstances(model);here!=NULL;
                        here=ISRCnextInstance(here)) {

/* check if the source has a distortion input*/

if (here->ISRCdGiven) {
  if (here->ISRCdF2given) job->Df2given = 1;
  if ((here->ISRCdF1given) && (mode == D_RHSF1)) {
 
     mag = here->ISRCdF1mag;
     phase = here->ISRCdF1phase;
}
else if ((here->ISRCdF2given) && (mode == D_RHSF2)) {

     mag = here->ISRCdF2mag;
     phase = here->ISRCdF2phase;
}
if (((here->ISRCdF1given) && (mode == D_RHSF1)) || 
	((here->ISRCdF2given) && (mode == D_RHSF2))) {
   
	ckt->CKTrhs[here->ISRCposNode] = - 0.5 * mag
		* cos(M_PI*phase/180.0);
        ckt->CKTrhs[here->ISRCnegNode] =   0.5 * mag * cos(
		M_PI*phase/180.0);
	ckt->CKTirhs[here->ISRCposNode] = - 0.5 * mag * sin(
		M_PI*phase/180.0);
	ckt->CKTirhs[here->ISRCnegNode] = 0.5 * mag * sin(
		M_PI*phase/180.0); 
}
                    }
                }
            }
        }
}
error = 0;
break;

    default: 

	    error = E_BADPARM;
	    break;
    }

    return(error);

}

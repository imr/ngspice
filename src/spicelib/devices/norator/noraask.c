/**********
Author: Florian Ballenegger 2020
Adapted from VCVS device code.
**********/
/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/

/*
 * This routine gives access to the internal device parameters
 * of Voltage Controlled Voltage Source
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "noradefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/* ARGSUSED */
int
NORAask(CKTcircuit *ckt, GENinstance *inst, int which, IFvalue *value, IFvalue *select)
{
    NORAinstance *here = (NORAinstance *)inst;
    double vr;
    double vi;
    double sr;
    double si;
    double vm;
    static char *msg = "Current and power not available for ac analysis";
    switch(which) {
        case NORA_POS_NODE:
            value->iValue = here->NORAposNode;
            return (OK);
        case NORA_NEG_NODE:
            value->iValue = here->NORAnegNode;
            return (OK);
        case NORA_IC:
            value->rValue = here->NORAinitCond;
            return (OK);
        case NORA_BR:
            value->iValue = here->NORAbranch;
            return (OK);
        case NORA_CURRENT :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = TMALLOC(char, strlen(msg) + 1);
                errRtn = "NORAask";
                strcpy(errMsg,msg);
                return(E_ASKCURRENT);
            } else {
                value->rValue = *(ckt->CKTrhsOld + here->NORAbranch);
            }
            return(OK);
        case NORA_VOLTS :
	    value->rValue = (*(ckt->CKTrhsOld + here->NORAposNode) - 
		*(ckt->CKTrhsOld + here->NORAnegNode));
            return(OK);
        case NORA_POWER :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = TMALLOC(char, strlen(msg) + 1);
                errRtn = "NORAask";
                strcpy(errMsg,msg);
                return(E_ASKPOWER);
            } else {
                value->rValue = *(ckt->CKTrhsOld + here->NORAbranch) *
                        (*(ckt->CKTrhsOld + here->NORAposNode) - 
                        *(ckt->CKTrhsOld + here->NORAnegNode));
            }
            return(OK);
        default:
            return (E_BADPARM);
    }
    /* NOTREACHED */
}

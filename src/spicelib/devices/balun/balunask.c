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
#include "balundefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/* ARGSUSED */
int
BALUNask(CKTcircuit *ckt, GENinstance *inst, int which, IFvalue *value, IFvalue *select)
{
    BALUNinstance *here = (BALUNinstance *)inst;
    static char *msg = "Power not available for ac analysis";
    switch(which) {
        case BALUN_POS_NODE:
            value->iValue = here->BALUNposNode;
            return (OK);
        case BALUN_NEG_NODE:
            value->iValue = here->BALUNnegNode;
            return (OK);
        case BALUN_DIFF_NODE:
            value->iValue = here->BALUNdiffNode;
            return (OK);
        case BALUN_CM_NODE:
            value->iValue = here->BALUNcmNode;
            return (OK);
        case BALUN_BRPOS:
            value->iValue = here->BALUNbranchpos;
            return (OK);
	case BALUN_BRNEG:
            value->iValue = here->BALUNbranchneg;
            return (OK);
	    
        case BALUN_POWER :
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = TMALLOC(char, strlen(msg) + 1);
                errRtn = "BALUNask";
                strcpy(errMsg,msg);
                return(E_ASKPOWER);
            } else {
                value->rValue = 
		        *(ckt->CKTrhsOld + here->BALUNbranchpos) *
                        (*(ckt->CKTrhsOld + here->BALUNposNode)) + 
			*(ckt->CKTrhsOld + here->BALUNbranchneg) *
                        (*(ckt->CKTrhsOld + here->BALUNnegNode));
            }
            return(OK);
        default:
            return (E_BADPARM);
    }
    /* NOTREACHED */
}

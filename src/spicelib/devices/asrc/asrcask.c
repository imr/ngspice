/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Kanwar Jit Singh
**********/
/*
 * singh@ic.Berkeley.edu
 */

/*
 * This routine gives access to the internal device parameters
 * of Current Controlled Voltage Source
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "ifsim.h"
#include "asrcdefs.h"
#include "sperror.h"
#include "suffix.h"


/* ARGSUSED */
int
ASRCask(CKTcircuit *ckt, GENinstance *instPtr, int which, IFvalue *value, IFvalue *select)
{
    ASRCinstance *here = (ASRCinstance*)instPtr;

    switch(which) {
        case ASRC_CURRENT:
            value->tValue = here->ASRCtype == ASRC_CURRENT ? 
                    here->ASRCtree : NULL;
            return (OK);
        case ASRC_VOLTAGE:
            value->tValue = here->ASRCtype == ASRC_VOLTAGE ? 
                    here->ASRCtree : NULL;
            return (OK);
        case ASRC_POS_NODE:
            value->iValue = here->ASRCposNode;
            return (OK);
        case ASRC_NEG_NODE:
            value->iValue = here->ASRCnegNode;
            return (OK);
        case ASRC_OUTPUTCURRENT:
	    value->rValue = ckt->CKTrhsOld[here->ASRCbranch];
            return (OK);
        case ASRC_OUTPUTVOLTAGE:
	    value->rValue = ckt->CKTrhsOld[here->ASRCposNode] -
		ckt->CKTrhsOld[here->ASRCnegNode];
	    return(OK);
        default:
            return (E_BADPARM);
    }
    /* NOTREACHED */
}

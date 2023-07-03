/**********
Copyright 1992 Regents of the University of California.  All rights
reserved.
Author: 2004 Paolo Nenzi
**********/


#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "cpldefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
CPLask(CKTcircuit *ckt, GENinstance *inst, int which, IFvalue *value, IFvalue *select)
{
    CPLinstance *here = (CPLinstance *)inst;

    NG_IGNORE(ckt);
    NG_IGNORE(select);

    switch(which) {
        /* String tables are expected to be freed, but not the strings. */

        case CPL_POS_NODE:
            value->v.vec.sVec = TMALLOC(char *, here->dimension);
            memcpy(value->v.vec.sVec, here->in_node_names,
                   here->dimension * sizeof (char *));
	    value->v.numValue = here->dimension;
            return(OK);
        case CPL_NEG_NODE:
            value->v.vec.sVec = TMALLOC(char *, here->dimension);
            memcpy(value->v.vec.sVec, here->out_node_names,
                   here->dimension * sizeof (char *));
	    value->v.numValue = here->dimension;
            return(OK);
        case CPL_DIM:
            value->iValue = here->dimension;
            return(OK);
	case CPL_LENGTH:
	    value->rValue = here->CPLlength;
            return(OK);
        default:
            return(E_BADPARM);
    }
}

/**********
Copyright 1992 Regents of the University of California.  All rights
reserved.
Author: 2004 Paolo Nenzi
**********/


#include "ngspice.h"
#include "const.h"
#include "ifsim.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "cpldefs.h"
#include "sperror.h"
#include "suffix.h"


int
CPLask(CKTcircuit *ckt, GENinstance *inst, int which, IFvalue *value, IFvalue *select)
{
    CPLinstance *here = (CPLinstance *)inst;

    switch(which) {
        case CPL_POS_NODE:
            value->v.vec.sVec = here->in_node_names;
	    value->v.numValue = here->dimension;
            return(OK);
        case CPL_NEG_NODE:
            value->v.vec.sVec = here->out_node_names;
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
    return(OK);
}

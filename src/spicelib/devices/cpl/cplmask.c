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

/*ARGSUSED*/
int
CPLmAsk(CKTcircuit *ckt, GENmodel *inModel, int which, IFvalue *value)
{
    CPLmodel *model = (CPLmodel *)inModel;
    switch(which) {
        case CPL_R:
            value->v.vec.rVec = model->Rm;
            value->v.numValue = model->Rm_counter;
            return(OK);
        case CPL_L:
            value->v.vec.rVec = model->Lm;
            value->v.numValue = model->Lm_counter;
            return(OK);
        case CPL_G:
            value->v.vec.rVec = model->Gm;
            value->v.numValue = model->Gm_counter;
            return(OK);
        case CPL_C:
            value->v.vec.rVec = model->Cm;
            value->v.numValue = model->Cm_counter;
            return(OK);
        case CPL_length:
            value->rValue = model->length;
            return(OK);
	case CPL_MOD_R:
            /* No op */
	    return(OK);
        default:
            return(E_BADPARM);
    }
     /* NOTREACHED */ 
}

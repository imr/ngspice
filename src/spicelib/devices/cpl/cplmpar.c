/**********
Copyright 1992 Regents of the University of California.  All rights
reserved.
Author: 1992 Charles Hough
**********/


#include "ngspice.h"
#include "const.h"
#include "ifsim.h"
#include "cpldefs.h"
#include "sperror.h"
#include "suffix.h"


int
CPLmParam(int param, IFvalue *value, GENmodel *inModel)
{
    register CPLmodel *model = (CPLmodel *)inModel;
    switch(param) {
        case CPL_R:
            model->Rm = value->v.vec.rVec;
            model->Rm_counter = value->v.numValue;
            model->Rmgiven = TRUE;
            break;
        case CPL_L:
            model->Lm = value->v.vec.rVec;
            model->Lm_counter = value->v.numValue;
            model->Lmgiven = TRUE;
            break;
        case CPL_G:
            model->Gm = value->v.vec.rVec;
            model->Gm_counter = value->v.numValue;
            model->Gmgiven = TRUE;
            break;
        case CPL_C:
            model->Cm = value->v.vec.rVec;
            model->Cm_counter = value->v.numValue;
            model->Cmgiven = TRUE;
            break;
        case CPL_length:
            model->length = value->rValue;
            model->lengthgiven = TRUE;
            break;
	case CPL_MOD_R:
	    break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}

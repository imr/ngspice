/**********
Copyright 1992 Regents of the University of California.  All rights
reserved.
Author: 1992 Charles Hough
**********/


#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "cpldefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "cplhash.h"


static void copy_coeffs(double **dst, IFvalue *value)
{
    int n = value->v.numValue;

    if (*dst) {
        tfree(*dst);
    }

    *dst = TMALLOC(double, n);

    memcpy(*dst, value->v.vec.rVec, (size_t) n * sizeof(double));
}


int
CPLmParam(int param, IFvalue *value, GENmodel *inModel)
{
    CPLmodel *model = (CPLmodel *)inModel;
    switch(param) {
        case CPL_R:
            copy_coeffs(& model->Rm, value);
            model->Rm_counter = value->v.numValue;
            model->Rmgiven = TRUE;
            break;
        case CPL_L:
            copy_coeffs(& model->Lm, value);
            model->Lm_counter = value->v.numValue;
            model->Lmgiven = TRUE;
            break;
        case CPL_G:
            copy_coeffs(& model->Gm, value);
            model->Gm_counter = value->v.numValue;
            model->Gmgiven = TRUE;
            break;
        case CPL_C:
            copy_coeffs(& model->Cm, value);
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

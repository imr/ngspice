/**********
Based on jfetmask.c
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Mathew Lew and Thomas L. Quarles

Modified to add PS model and new parameter definitions ( Anthony E. Parker )
   Copyright 1994  Macquarie University, Sydney Australia.
   10 Feb 1994: Added call to jfetparm.h
**********/
/*
 */

#include "ngspice.h"
#include "const.h"
#include "ifsim.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "jfet2defs.h"
#include "sperror.h"
#include "suffix.h"


/*ARGSUSED*/
int
JFET2mAsk(CKTcircuit *ckt, GENmodel *inModel, int which, IFvalue *value)
{
    JFET2model *model = (JFET2model*)inModel;
    switch(which) {
        case JFET2_MOD_TNOM:
            value->rValue = model->JFET2tnom-CONSTCtoK;
            return(OK);
 
#define PARAM(code,id,flag,ref,default,descrip) case id: \
                                   value->rValue = model->ref; return(OK);
#include "jfet2parm.h"
 
        case JFET2_MOD_DRAINCONDUCT:
            value->rValue = model->JFET2drainConduct;
            return(OK);
        case JFET2_MOD_SOURCECONDUCT:
            value->rValue = model->JFET2sourceConduct;
            return(OK);
        case JFET2_MOD_TYPE:
	    if (model->JFET2type == NJF)
                value->sValue = "njf";
	    else
                value->sValue = "pjf";
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}


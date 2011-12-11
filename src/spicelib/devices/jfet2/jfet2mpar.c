/**********
Based on jfetmpar.c
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

Modified to add PS model and new parameter definitions ( Anthony E. Parker )
   Copyright 1994  Macquarie University, Sydney Australia.
   10 Feb 1994: Added call to jfetparm.h
**********/

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "jfet2defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
JFET2mParam(int param, IFvalue *value, GENmodel *inModels)
{
    JFET2model *model = (JFET2model*)inModels;
    switch(param) {
        case JFET2_MOD_TNOM:
            model->JFET2tnomGiven = TRUE;
            model->JFET2tnom = value->rValue+CONSTCtoK;
            break;
#define PARAM(code,id,flag,ref,default,descrip) case id: \
                      model->flag = TRUE; model->ref = value->rValue; break;
#include "jfet2parm.h"
        case JFET2_MOD_NJF:
            if(value->iValue) {
                model->JFET2type = NJF;
            }
            break;
        case JFET2_MOD_PJF:
            if(value->iValue) {
                model->JFET2type = PJF;
            }
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
